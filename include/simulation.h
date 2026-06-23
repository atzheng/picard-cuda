#pragma once
#include <cuda_runtime.h>
#include "types.h"
#include "rng.h"
#include "fulfillment.h"

// simulate_worker_kernel: each block is one worker, processing num_steps events sequentially.
// This is the CUDA equivalent of vmap(simulate_product) where each vmap lane runs lax.scan(step_in_time).
//
// Inputs (all device pointers):
//   nn_params, nn_layer_sizes, nn_num_layers: neural network
//   worker_inventory: [n_workers, max_products_per_worker, n_nodes_plus1]
//   worker_capacity:  [n_nodes_plus1] (shared across workers)
//   events_product:   [n_workers, num_steps] — product index within this worker
//   events_quantity:   [n_workers, num_steps]
//   events_node_near_to_far: [n_workers, num_steps, n_nodes_plus1]
//   events_capacity_delta: [n_workers, num_steps, n_nodes_plus1]
//   valid_mask:        [n_workers, num_steps] — 1 if event is valid
//   rng_keys:         [n_workers]
//
// Output:
//   fulfill_2D:       [n_workers, num_steps] — assigned node for each event

// Maximum sizes for stack allocation in the kernel
// Adjust these if layer sizes change
#define MAX_NN_DIM 128   // max(layer_sizes) — enough for [30,64,64,31]
#define MAX_NODES 64     // max n_nodes_plus1

__global__ void simulate_worker_kernel(
    // Neural net
    const float* nn_params,
    const int* nn_layer_sizes,
    int nn_num_layers,
    float greedy_cost_prob,
    // Worker state
    float* worker_inventory,    // [n_workers, max_ppw, n_nodes_plus1]
    const float* init_capacity, // [n_nodes_plus1]
    // Events for this iteration
    const int* events_product,           // [n_workers, num_steps]
    const int* events_quantity,          // [n_workers, num_steps]
    const int* events_node_near_to_far,  // [n_workers, num_steps, n_nodes_plus1]
    const float* events_capacity_delta,  // [n_workers, num_steps, n_nodes_plus1]
    const int* valid_mask,               // [n_workers, num_steps]
    const uint64_t* rng_keys,            // [n_workers]
    // Dimensions
    int n_workers,
    int num_steps,
    int max_ppw,
    int n_nodes_plus1,
    // Output
    int* fulfill_2D                      // [n_workers, num_steps]
) {
    int worker_id = blockIdx.x;
    if (worker_id >= n_workers) return;

    // Only thread 0 in each block does the work (sequential scan)
    if (threadIdx.x != 0) return;

    // Local capacity state (copy from init_capacity)
    float local_capacity[MAX_NODES];
    for (int n = 0; n < n_nodes_plus1; n++) {
        local_capacity[n] = init_capacity[n];
    }

    // Scratch buffers for NN inference
    float scratch_a[MAX_NN_DIM];
    float scratch_b[MAX_NN_DIM];
    float nn_output[MAX_NN_DIM];

    uint64_t key = rng_keys[worker_id];

    float* inv = worker_inventory + (size_t)worker_id * max_ppw * n_nodes_plus1;

    for (int step = 0; step < num_steps; step++) {
        int flat_idx = worker_id * num_steps + step;

        if (!valid_mask[flat_idx]) {
            fulfill_2D[flat_idx] = n_nodes_plus1 - 1;  // unfulfill
            continue;
        }

        // Apply capacity delta from other threads
        const float* cap_delta = events_capacity_delta + (size_t)flat_idx * n_nodes_plus1;
        for (int n = 0; n < n_nodes_plus1; n++) {
            local_capacity[n] -= cap_delta[n];
        }

        int product = events_product[flat_idx];
        int quantity = events_quantity[flat_idx];
        const int* node_ntf = events_node_near_to_far + (size_t)flat_idx * n_nodes_plus1;

        // Inventory row for this product within this worker
        float* inv_row = inv + (size_t)product * n_nodes_plus1;

        // Split key
        uint64_t subkey;
        split_key(key, key, subkey);

        // Run fulfillment
        int node = fulfillment(
            subkey, greedy_cost_prob,
            nn_params, nn_layer_sizes, nn_num_layers,
            inv_row, local_capacity, quantity,
            node_ntf, n_nodes_plus1,
            scratch_a, scratch_b, nn_output);

        fulfill_2D[flat_idx] = node;

        // Update inventory and capacity
        inv_row[node] -= (float)quantity;
        local_capacity[node] -= (float)quantity;
    }
}

// ============================================================
// Warp-cooperative kernel (P8, Variant B)
// ============================================================
//
// One WARP (32 lanes) processes one worker. The step loop stays serial (state is
// carried across steps), but within each event the expensive MLP matvec is split
// across lanes, and the small norm/logsumexp reductions are done with warp
// shuffles. The warp reductions sum floats in a different order than the
// single-thread kernel, so the output is NOT bit-identical — it is verified
// against the sequential scan with a node-decision match tolerance instead.

#define WARPS_PER_BLOCK 8   // 256 threads/block

__device__ inline float warp_reduce_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;  // lane 0 holds the total
}
__device__ inline float warp_reduce_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
    return v;  // lane 0 holds the max
}

__global__ void simulate_worker_kernel_warp(
    const float* nn_params,
    const int* nn_layer_sizes,
    int nn_num_layers,
    float greedy_cost_prob,
    float* worker_inventory,
    const float* init_capacity,
    const int* events_product,
    const int* events_quantity,
    const int* events_node_near_to_far,
    const float* events_capacity_delta,
    const int* valid_mask,
    const uint64_t* rng_keys,
    int n_workers,
    int num_steps,
    int max_ppw,
    int n_nodes_plus1,
    int* fulfill_2D
) {
    const int lane = threadIdx.x & 31;
    const int wpb  = threadIdx.x >> 5;
    const int worker_id = blockIdx.x * WARPS_PER_BLOCK + wpb;
    if (worker_id >= n_workers) return;  // uniform across the warp

    // Per-warp shared state and NN scratch (ping-pong buffers a/b, output, cap).
    __shared__ float s_cap[WARPS_PER_BLOCK][MAX_NODES];
    __shared__ float s_a[WARPS_PER_BLOCK][MAX_NN_DIM];
    __shared__ float s_b[WARPS_PER_BLOCK][MAX_NN_DIM];
    __shared__ float s_out[WARPS_PER_BLOCK][MAX_NN_DIM];
    __shared__ int   s_node[WARPS_PER_BLOCK];

    float* cap = s_cap[wpb];
    for (int n = lane; n < n_nodes_plus1; n += 32) cap[n] = init_capacity[n];
    __syncwarp();

    uint64_t key = rng_keys[worker_id];
    float* inv = worker_inventory + (size_t)worker_id * max_ppw * n_nodes_plus1;

    const int in_size  = nn_layer_sizes[0];
    const int out_size = nn_layer_sizes[nn_num_layers];

    for (int step = 0; step < num_steps; step++) {
        int flat_idx = worker_id * num_steps + step;

        if (!valid_mask[flat_idx]) {
            if (lane == 0) fulfill_2D[flat_idx] = n_nodes_plus1 - 1;
            continue;
        }

        const float* cap_delta = events_capacity_delta + (size_t)flat_idx * n_nodes_plus1;
        for (int n = lane; n < n_nodes_plus1; n += 32) cap[n] -= cap_delta[n];
        __syncwarp();

        int product  = events_product[flat_idx];
        int quantity = events_quantity[flat_idx];
        const int* node_ntf = events_node_near_to_far + (size_t)flat_idx * n_nodes_plus1;
        float* inv_row = inv + (size_t)product * n_nodes_plus1;

        uint64_t subkey;
        split_key(key, key, subkey);                 // identical on every lane
        bool use_cost = rand_bernoulli(subkey, greedy_cost_prob);

        if (use_cost) {
            // --- MLP forward pass, outputs split across lanes ---
            float* a = s_a[wpb];
            float* b = s_b[wpb];
            for (int i = lane; i < in_size; i += 32) a[i] = inv_row[i];
            __syncwarp();

            int param_offset = 0;
            for (int layer = 0; layer < nn_num_layers; layer++) {
                int nin  = nn_layer_sizes[layer];
                int nout = nn_layer_sizes[layer + 1];
                const float* W = nn_params + param_offset;
                const float* bias = W + nout * nin;
                param_offset += nout * nin + nout;
                bool last = (layer == nn_num_layers - 1);
                float* dst = last ? s_out[wpb] : b;
                // Each lane computes a disjoint set of outputs; the inner dot is
                // serial per output, so its summation order matches the scalar
                // kernel exactly (only the later reductions differ).
                for (int i = lane; i < nout; i += 32) {
                    float sum = bias[i];
                    for (int j = 0; j < nin; j++) sum += W[i * nin + j] * a[j];
                    dst[i] = last ? sum : fmaxf(0.0f, sum);
                }
                __syncwarp();
                if (!last) { float* t = a; a = b; b = t; }  // swap ping-pong
            }

            // --- logsumexp normalize over s_out[0..out_size) (warp reductions) ---
            float* out = s_out[wpb];
            float m = -1e30f;
            for (int i = lane; i < out_size; i += 32) m = fmaxf(m, out[i]);
            m = warp_reduce_max(m);
            m = __shfl_sync(0xffffffffu, m, 0);
            float se = 0.0f;
            for (int i = lane; i < out_size; i += 32) se += expf(out[i] - m);
            se = warp_reduce_sum(se);
            se = __shfl_sync(0xffffffffu, se, 0);
            float lse = m + logf(se);
            for (int i = lane; i < out_size; i += 32) out[i] -= lse;
            __syncwarp();

            // --- norm of the (normalized) output ---
            float nr = 0.0f;
            for (int i = lane; i < n_nodes_plus1; i += 32) nr += out[i] * out[i];
            nr = warp_reduce_sum(nr);
            nr = __shfl_sync(0xffffffffu, nr, 0);
            nr = sqrtf(nr + 1e-12f);

            // --- feasibility scan (cheap, serial on lane 0) ---
            if (lane == 0) {
                int chosen = node_ntf[n_nodes_plus1 - 1];
                for (int rank = 0; rank < n_nodes_plus1; rank++) {
                    int node = node_ntf[rank];
                    float iv = inv_row[node] + roundf(1e-7f * out[rank] / nr);
                    if (cap[node] >= (float)quantity && iv >= (float)quantity) {
                        chosen = node; break;
                    }
                }
                s_node[wpb] = chosen;
            }
        } else {
            // greedy_capacity: argmax feasible capacity (serial on lane 0)
            if (lane == 0) {
                int best_node = -1; float best_cap = -1.0f;
                for (int rank = 0; rank < n_nodes_plus1 - 1; rank++) {
                    int node = node_ntf[rank];
                    if (cap[node] >= (float)quantity && inv_row[node] >= (float)quantity
                        && cap[node] > best_cap) {
                        best_cap = cap[node]; best_node = node;
                    }
                }
                s_node[wpb] = (best_node < 0) ? node_ntf[n_nodes_plus1 - 1] : best_node;
            }
        }
        __syncwarp();
        int node = s_node[wpb];

        if (lane == 0) {
            fulfill_2D[flat_idx] = node;
            inv_row[node] -= (float)quantity;
            cap[node] -= (float)quantity;
        }
        __syncwarp();
    }
}

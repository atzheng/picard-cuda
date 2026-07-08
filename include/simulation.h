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
    const double* nn_params,
    const int* nn_layer_sizes,
    int nn_num_layers,
    double greedy_cost_prob,
    // Worker state. inventory_scratch is the full [n_products, n_nodes_plus1]
    // inventory (a fresh per-iteration copy). Products are partitioned 1:1 to
    // workers, so each worker only ever touches rows for its own products — the
    // writes are disjoint across workers, no per-worker private copy needed (P13).
    double* inventory_scratch,   // [n_products, n_nodes_plus1]
    const double* init_capacity, // [n_nodes_plus1]
    // Events for this iteration
    const int* events_product,           // [n_workers, num_steps] — original product id
    const int* events_quantity,          // [n_workers, num_steps]
    const int* event_ntf_full,           // [n_events, n_nodes_plus1] — immutable, device-resident
    const int* order_2D_index,           // [n_workers, num_steps] — global event idx per slot (-1 if none)
    const double* events_capacity_delta,  // [n_workers, num_steps, n_nodes_plus1]
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
    double local_capacity[MAX_NODES];
    for (int n = 0; n < n_nodes_plus1; n++) {
        local_capacity[n] = init_capacity[n];
    }

    // Scratch buffers for NN inference
    double scratch_a[MAX_NN_DIM];
    double scratch_b[MAX_NN_DIM];
    double nn_output[MAX_NN_DIM];

    uint64_t key = rng_keys[worker_id];

    for (int step = 0; step < num_steps; step++) {
        int flat_idx = worker_id * num_steps + step;

        if (!valid_mask[flat_idx]) {
            fulfill_2D[flat_idx] = n_nodes_plus1 - 1;  // unfulfill
            continue;
        }

        // Apply capacity delta from other threads
        const double* cap_delta = events_capacity_delta + (size_t)flat_idx * n_nodes_plus1;
        for (int n = 0; n < n_nodes_plus1; n++) {
            local_capacity[n] -= cap_delta[n];
        }

        int product = events_product[flat_idx];
        int quantity = events_quantity[flat_idx];
        // Index the immutable, device-resident full ntf by this slot's global event
        // index (valid here: the !valid_mask branch above already `continue`d).
        const int* node_ntf = event_ntf_full + (size_t)order_2D_index[flat_idx] * n_nodes_plus1;

        // Inventory row for this product (rows are disjoint across workers).
        double* inv_row = inventory_scratch + (size_t)product * n_nodes_plus1;

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
        inv_row[node] -= (double)quantity;
        local_capacity[node] -= (double)quantity;
    }
}

// ============================================================
// Warp-cooperative kernel (P8, Variant B)
// ============================================================
//
// One WARP (32 lanes) processes one worker. The step loop stays serial (state is
// carried across steps), but within each event the expensive MLP matvec is split
// across lanes, and the small norm/logsumexp reductions are done with warp
// shuffles. The warp reductions sum in a different order than the single-thread
// kernel, so the output is NOT bit-identical — it is verified against the
// sequential scan with a node-decision match tolerance instead.

#define WARPS_PER_BLOCK 8   // 256 threads/block

__device__ inline double warp_reduce_sum(double v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;  // lane 0 holds the total
}
__device__ inline double warp_reduce_max(double v) {
    for (int o = 16; o > 0; o >>= 1) v = fmax(v, __shfl_down_sync(0xffffffffu, v, o));
    return v;  // lane 0 holds the max
}

__global__ void simulate_worker_kernel_warp(
    const double* nn_params,
    const int* nn_layer_sizes,
    int nn_num_layers,
    double greedy_cost_prob,
    double* inventory_scratch,            // [n_products, n_nodes_plus1] — disjoint rows per worker (P13)
    const double* init_capacity,
    const int* events_product,           // original product id per slot
    const int* events_quantity,
    const int* event_ntf_full,           // [n_events, n_nodes_plus1] — immutable, device-resident
    const int* order_2D_index,           // [n_workers, num_steps] — global event idx per slot
    const double* events_capacity_delta,
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
    __shared__ double s_cap[WARPS_PER_BLOCK][MAX_NODES];
    __shared__ double s_a[WARPS_PER_BLOCK][MAX_NN_DIM];
    __shared__ double s_b[WARPS_PER_BLOCK][MAX_NN_DIM];
    __shared__ double s_out[WARPS_PER_BLOCK][MAX_NN_DIM];
    __shared__ int   s_node[WARPS_PER_BLOCK];

    double* cap = s_cap[wpb];
    for (int n = lane; n < n_nodes_plus1; n += 32) cap[n] = init_capacity[n];
    __syncwarp();

    uint64_t key = rng_keys[worker_id];

    const int in_size  = nn_layer_sizes[0];
    const int out_size = nn_layer_sizes[nn_num_layers];

    for (int step = 0; step < num_steps; step++) {
        int flat_idx = worker_id * num_steps + step;

        if (!valid_mask[flat_idx]) {
            if (lane == 0) fulfill_2D[flat_idx] = n_nodes_plus1 - 1;
            continue;
        }

        const double* cap_delta = events_capacity_delta + (size_t)flat_idx * n_nodes_plus1;
        for (int n = lane; n < n_nodes_plus1; n += 32) cap[n] -= cap_delta[n];
        __syncwarp();

        int product  = events_product[flat_idx];
        int quantity = events_quantity[flat_idx];
        // Index the immutable, device-resident full ntf by this slot's global event
        // index (valid here: the !valid_mask branch above already `continue`d).
        const int* node_ntf = event_ntf_full + (size_t)order_2D_index[flat_idx] * n_nodes_plus1;
        double* inv_row = inventory_scratch + (size_t)product * n_nodes_plus1;

        uint64_t subkey;
        split_key(key, key, subkey);                 // identical on every lane
        bool use_cost = rand_bernoulli(subkey, greedy_cost_prob);

        if (use_cost) {
            // --- MLP forward pass, outputs split across lanes ---
            double* a = s_a[wpb];
            double* b = s_b[wpb];
            for (int i = lane; i < in_size; i += 32) a[i] = inv_row[i];
            __syncwarp();

            int param_offset = 0;
            for (int layer = 0; layer < nn_num_layers; layer++) {
                int nin  = nn_layer_sizes[layer];
                int nout = nn_layer_sizes[layer + 1];
                const double* W = nn_params + param_offset;
                const double* bias = W + nout * nin;
                param_offset += nout * nin + nout;
                bool last = (layer == nn_num_layers - 1);
                double* dst = last ? s_out[wpb] : b;
                // Each lane computes a disjoint set of outputs; the inner dot is
                // serial per output, so its summation order matches the scalar
                // kernel exactly (only the later reductions differ).
                for (int i = lane; i < nout; i += 32) {
                    double sum = bias[i];
                    for (int j = 0; j < nin; j++) sum += W[i * nin + j] * a[j];
                    dst[i] = last ? sum : fmax(0.0, sum);
                }
                __syncwarp();
                if (!last) { double* t = a; a = b; b = t; }  // swap ping-pong
            }

            // --- logsumexp normalize over s_out[0..out_size) (warp reductions) ---
            double* out = s_out[wpb];
            double m = -1e30;
            for (int i = lane; i < out_size; i += 32) m = fmax(m, out[i]);
            m = warp_reduce_max(m);
            m = __shfl_sync(0xffffffffu, m, 0);
            double se = 0.0;
            for (int i = lane; i < out_size; i += 32) se += exp(out[i] - m);
            se = warp_reduce_sum(se);
            se = __shfl_sync(0xffffffffu, se, 0);
            double lse = m + log(se);
            for (int i = lane; i < out_size; i += 32) out[i] -= lse;
            __syncwarp();

            // --- norm of the (normalized) output ---
            double nr = 0.0;
            for (int i = lane; i < n_nodes_plus1; i += 32) nr += out[i] * out[i];
            nr = warp_reduce_sum(nr);
            nr = __shfl_sync(0xffffffffu, nr, 0);
            nr = sqrt(nr + 1e-12);

            // --- feasibility scan (cheap, serial on lane 0) ---
            if (lane == 0) {
                int chosen = node_ntf[n_nodes_plus1 - 1];
                for (int rank = 0; rank < n_nodes_plus1; rank++) {
                    int node = node_ntf[rank];
                    double iv = inv_row[node] + round(1e-7 * out[rank] / nr);
                    if (cap[node] >= (double)quantity && iv >= (double)quantity) {
                        chosen = node; break;
                    }
                }
                s_node[wpb] = chosen;
            }
        } else {
            // greedy_capacity: argmax feasible capacity (serial on lane 0)
            if (lane == 0) {
                int best_node = -1; double best_cap = -1.0;
                for (int rank = 0; rank < n_nodes_plus1 - 1; rank++) {
                    int node = node_ntf[rank];
                    if (cap[node] >= (double)quantity && inv_row[node] >= (double)quantity
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
            inv_row[node] -= (double)quantity;
            cap[node] -= (double)quantity;
        }
        __syncwarp();
    }
}

// ============================================================
// GPU capacity-delta (P9 prep on device)
// ============================================================
//
// Moves compute_capacity_delta_other_threads onto the GPU. capacity_delta is the
// largest per-window structure (evc·np1) and was both computed on the host (a
// 31M-element cumsum + gather, the steady-state prep bottleneck) and uploaded H2D
// (124 MB). Quantities are all 1, so every cumsum partial is an exact integer
// (|·| ≤ eff ≪ 2^24) — double adds are exact regardless of order — so the device
// result is **bit-identical** to the host version.

// Per-column prefix sum: cumsum[t,n] = -count(s<=t : fulfill[s]==n) (×quantity).
// One BLOCK per node-column (grid = np1); the block scans its column in chunks of
// blockDim with a running carry, each chunk a parallel block-inclusive scan
// (warp-shuffle scan + a serial scan over the per-warp totals). Integer partials, so
// the result is exact regardless of the intra-chunk reduction order — bit-identical
// to the host serial scan. blockDim must be a multiple of 32 (use 256). Launch
// <<<np1, 256>>>.
__global__ void capacity_cumsum_kernel(
    const int* __restrict__ fulfill,    // [eff]
    const int* __restrict__ quant,      // [eff]
    int eff, int np1,
    double* __restrict__ cumsum         // [eff, np1]
) {
    const int n = blockIdx.x;
    if (n >= np1) return;
    const int T = blockDim.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = T >> 5;
    __shared__ double wsum[32];   // per-warp totals
    __shared__ double wexcl[32];  // exclusive prefix of the warp totals
    __shared__ double carry_s;    // running column total across chunks
    __shared__ double btot_s;     // this chunk's block total
    if (threadIdx.x == 0) carry_s = 0.0;
    __syncthreads();

    for (int base = 0; base < eff; base += T) {
        int t = base + threadIdx.x;
        double x = (t < eff && fulfill[t] == n) ? -(double)quant[t] : 0.0;
        // Warp-inclusive scan.
        for (int o = 1; o < 32; o <<= 1) {
            double y = __shfl_up_sync(0xffffffffu, x, o);
            if (lane >= o) x += y;
        }
        if (lane == 31) wsum[warp] = x;
        __syncthreads();
        // Serial exclusive scan over the (≤32) per-warp totals.
        if (threadIdx.x == 0) {
            double acc = 0.0;
            for (int w = 0; w < nwarps; w++) { wexcl[w] = acc; acc += wsum[w]; }
            btot_s = acc;
        }
        __syncthreads();
        if (t < eff) cumsum[(size_t)t * np1 + n] = carry_s + x + wexcl[warp];
        __syncthreads();
        if (threadIdx.x == 0) carry_s += btot_s;
        __syncthreads();
    }
}

// Gather: capacity_delta[w,s,n] = -((caps[w,s,n] - caps[w,s-1,n]) - by_product),
// caps[w,s,n] = cumsum[order_rel[w,s], n] (0 if the slot is empty). One thread per
// output element (tid = (w·num_steps+s)·np1 + n); writes and the cumsum reads for a
// warp (fixed slot, consecutive n) are coalesced. Mirrors the host gather exactly.
__global__ void capacity_gather_kernel(
    const double* __restrict__ cumsum,  // [eff, np1]
    const int* __restrict__ order_rel,  // [n_workers*num_steps], in [0,eff) or -1
    const int* __restrict__ fulfill,    // [eff]
    const int* __restrict__ quant,      // [eff]
    int n_workers, int num_steps, int eff, int np1,
    double* __restrict__ cap_delta      // [n_workers*num_steps, np1]
) {
    long tid = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long total = (long)n_workers * num_steps * np1;
    if (tid >= total) return;
    int n  = (int)(tid % np1);
    long ws = tid / np1;                 // worker*num_steps + s
    int s  = (int)(ws % num_steps);

    int idx = order_rel[ws];
    double cap_val = (idx >= 0 && idx < eff) ? cumsum[(size_t)idx * np1 + n] : 0.0;
    double cap_prev = 0.0;
    if (s > 0) {
        int pidx = order_rel[ws - 1];
        if (pidx >= 0 && pidx < eff) cap_prev = cumsum[(size_t)pidx * np1 + n];
    }
    double cd = cap_val - cap_prev;
    double by_product = (idx >= 0 && idx < eff && fulfill[idx] == n) ? -(double)quant[idx] : 0.0;
    cap_delta[tid] = -(cd - by_product);
}

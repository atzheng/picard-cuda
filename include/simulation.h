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

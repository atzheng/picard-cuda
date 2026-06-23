#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include "rng.h"
#include "neural_net.h"

// Greedy-cost fulfillment: pick first feasible node in near-to-far order,
// with NN output as infinitesimal tie-breaker on inventory.
// This is the default policy from jax-sim-v2.py (_fulfillment_greedy_cost).
__device__ inline int fulfillment_greedy_cost(
    const float* nn_params,
    const int* nn_layer_sizes,
    int nn_num_layers,
    const float* inventory_row,    // [n_nodes_plus1] for this product
    const float* capacity,         // [n_nodes_plus1]
    int quantity,
    const int* node_near_to_far,   // [n_nodes_plus1] sorted node indices
    int n_nodes_plus1,
    float* scratch_a,              // scratch buffer for NN: max(layer_sizes)
    float* scratch_b,              // scratch buffer for NN: max(layer_sizes)
    float* nn_output               // scratch buffer for NN output: layer_sizes[last]
) {
    int n_nodes = n_nodes_plus1 - 1;  // exclude unfulfill sentinel

    // Run NN on inventory (excluding unfulfill sentinel column)
    nn_predict(nn_params, nn_layer_sizes, nn_num_layers,
               inventory_row, nn_output, scratch_a, scratch_b);

    // Compute norm of NN output
    float norm = 0.0f;
    for (int i = 0; i < n_nodes_plus1; i++) {
        norm += nn_output[i] * nn_output[i];
    }
    norm = sqrtf(norm + 1e-12f);

    // For each node in near-to-far order, check feasibility
    // inventory is tie-broken with 1e-7 * output / norm
    for (int rank = 0; rank < n_nodes_plus1; rank++) {
        int node = node_near_to_far[rank];
        float inv = inventory_row[node] + roundf(1e-7f * nn_output[rank] / norm);
        float cap = capacity[node];
        if (cap >= (float)quantity && inv >= (float)quantity) {
            return node;
        }
    }
    // Unfulfill: return sentinel node (last index)
    return node_near_to_far[n_nodes_plus1 - 1];
}

// Greedy-capacity fulfillment: pick feasible node with maximum remaining capacity.
__device__ inline int fulfillment_greedy_capacity(
    const float* inventory_row,    // [n_nodes_plus1] for this product
    const float* capacity,         // [n_nodes_plus1]
    int quantity,
    const int* node_near_to_far,   // [n_nodes_plus1]
    int n_nodes_plus1
) {
    int best_node = -1;
    float best_cap = -1.0f;

    // Exclude sentinel (last entry) from argmax
    for (int rank = 0; rank < n_nodes_plus1 - 1; rank++) {
        int node = node_near_to_far[rank];
        float inv = inventory_row[node];
        float cap = capacity[node];
        if (cap >= (float)quantity && inv >= (float)quantity) {
            if (cap > best_cap) {
                best_cap = cap;
                best_node = node;
            }
        }
    }

    if (best_node < 0) {
        // No feasible real node — unfulfill
        return node_near_to_far[n_nodes_plus1 - 1];
    }
    return best_node;
}

// Combined fulfillment: interpolate between greedy_cost and greedy_capacity
// using nn.greedy_cost_prob (Bernoulli draw).
__device__ inline int fulfillment(
    uint64_t& rng_key,
    float greedy_cost_prob,
    const float* nn_params,
    const int* nn_layer_sizes,
    int nn_num_layers,
    const float* inventory_row,
    const float* capacity,
    int quantity,
    const int* node_near_to_far,
    int n_nodes_plus1,
    float* scratch_a,
    float* scratch_b,
    float* nn_output
) {
    // In jax-sim-v2.py, fulfillment always calls _fulfillment_greedy_cost
    // (the greedy_cost_prob / greedy_capacity branch is in v2's _fulfillment_greedy)
    // But we implement both and select based on greedy_cost_prob
    bool use_cost = rand_bernoulli(rng_key, greedy_cost_prob);

    if (use_cost) {
        return fulfillment_greedy_cost(
            nn_params, nn_layer_sizes, nn_num_layers,
            inventory_row, capacity, quantity,
            node_near_to_far, n_nodes_plus1,
            scratch_a, scratch_b, nn_output);
    } else {
        return fulfillment_greedy_capacity(
            inventory_row, capacity, quantity,
            node_near_to_far, n_nodes_plus1);
    }
}

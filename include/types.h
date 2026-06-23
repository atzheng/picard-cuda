#pragma once
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

// All arrays use float32 for numerical stability
// Index arrays use int32

struct NeuralNet {
    // Stored as flat arrays on device; layer_offsets describes layout
    // Each layer: weights (nout x nin) followed by biases (nout)
    float* params;          // device pointer to flat parameter buffer
    int* layer_sizes;       // host: e.g. {30, 64, 64, 31}
    int num_layers;         // number of weight layers (len(sizes) - 1)
    int total_param_count;  // total floats in params buffer

    float greedy_cost_prob; // probability of using greedy-cost vs greedy-capacity
};

struct Event {
    // All arrays have shape [n_events] or [n_events, n_nodes+1]
    int* product;                    // [n_events]
    int* quantity;                   // [n_events]
    int* node_index_near_to_far;     // [n_events, n_nodes+1]
    int n_events;
    int n_nodes_plus1;               // n_nodes + 1 (includes unfulfill sentinel)
};

struct WorkerState {
    float* inventory;   // [max_products_per_worker, n_nodes+1]
    float* capacity;    // [n_nodes+1]
    uint64_t rng_key;
};

struct AlgoState {
    int iteration;
    int t_reset;
    int n_conflicts;
    float* capacity;    // [n_nodes+1] device
    float* inventory;   // [n_products, n_nodes+1] device
    int* fulfill;       // [n_events] device
    uint64_t rng_key;

    int n_products;
    int n_nodes_plus1;
    int n_events;
};

// Host-side problem description loaded from numpy files
struct ProblemData {
    // inventory[product, node] — shape [n_products, n_nodes]
    std::vector<float> inventory;
    // capacity[node] — shape [n_nodes] (single day)
    std::vector<float> capacity;
    // order products — shape [n_events]
    std::vector<int> order_products;
    // node_index_near_to_far[order, node_rank] — shape [n_events, n_nodes]
    std::vector<int> node_index_near_to_far;

    int n_products;
    int n_nodes;
    int n_events;
};

// Algorithm configuration
struct AlgoConfig {
    int max_workers;
    int max_products_per_worker;
    int num_steps;
    int max_iters;
    std::vector<int> layer_sizes;  // e.g. {30, 64, 64, 31}
    float greedy_cost_prob;
    int seed;
};

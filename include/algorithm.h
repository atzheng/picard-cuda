#pragma once
#include "types.h"

// Host-side functions that orchestrate GPU kernels

// Run one iteration of the algorithm (equivalent to iterate_algorithm in JAX)
// Modifies algo_state in-place on the host, launches kernels as needed.
void iterate_algorithm(
    AlgoState& algo_state,
    const Event& events,
    const NeuralNet& nn,
    const AlgoConfig& config
);

// Run the full algorithm for one day (equivalent to run_algorithm_day)
// while (iteration < max_iters && t_reset < n_events): iterate_algorithm
void run_algorithm_day(
    AlgoState& algo_state,
    const Event& events,
    const NeuralNet& nn,
    const AlgoConfig& config
);

// Allocate and initialize AlgoState on device from ProblemData
AlgoState create_algo_state(
    const ProblemData& problem,
    const AlgoConfig& config,
    uint64_t seed
);

// Free device memory in AlgoState
void free_algo_state(AlgoState& state);

// Allocate Event on device from ProblemData (single day)
Event create_events(const ProblemData& problem);

// Free device memory in Event
void free_events(Event& events);

// Allocate NeuralNet on device
NeuralNet create_neural_net(const AlgoConfig& config, uint64_t seed);

// Free device memory in NeuralNet
void free_neural_net(NeuralNet& nn);

// Copy fulfill array from device to host
void get_fulfill(const AlgoState& state, int* host_fulfill);

// Run the sequential simulation for one day (no Picard iteration, no workers).
// Processes all events in order, greedily assigning each to the best feasible node.
// This is the ground-truth baseline: equivalent to simulate_product with a single
// worker processing all events, with capacity_delta_other_threads = 0.
void run_sequential_day(
    AlgoState& algo_state,
    const Event& events,
    const NeuralNet& nn,
    const AlgoConfig& config
);

// Benchmark: run n_steps events with three variants and report timings.
// 1. dynamics-only:  greedy nearest-node (no NN), update inventory/capacity
// 2. nn-only:        run NN inference per event, ignore output, no state update
// 3. full:           NN inference + greedy fulfillment + state update
void run_benchmark(
    const AlgoState& algo_state,
    const Event& events,
    const NeuralNet& nn,
    int n_steps
);

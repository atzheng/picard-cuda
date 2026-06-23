#pragma once
#include "types.h"
#include <string>

// Load a ProblemData from a directory containing .npy files:
//   inventory.npy     — float64 [n_products, n_nodes]
//   capacity.npy      — float64 [n_days, n_nodes]  (we take day 0)
//   order_products.npy — int64 [n_events]
//   node_index_near_to_far.npy — int64 [n_events, n_nodes]
//
// These must be pre-exported from the Python NewProblem dataclass.
ProblemData load_problem(const std::string& directory, int day = 0);

#pragma once
#include <cuda_runtime.h>
#include <cstring>
#include <algorithm>
#include <cmath>

// ---- Device utility functions ----

// first_nonzero: return index of first true element, or fill if none
__device__ inline int first_nonzero_dev(const int* mask, int n, int fill) {
    for (int i = 0; i < n; i++) {
        if (mask[i]) return i;
    }
    return fill;
}

// Overload for bool arrays
__device__ inline int first_nonzero_dev(const bool* mask, int n, int fill) {
    for (int i = 0; i < n; i++) {
        if (mask[i]) return i;
    }
    return fill;
}

// ---- Host utility functions ----

// Stable argsort (ascending)
inline void argsort(const int* data, int* indices, int n) {
    for (int i = 0; i < n; i++) indices[i] = i;
    std::stable_sort(indices, indices + n, [data](int a, int b) {
        return data[a] < data[b];
    });
}

// cum_count: for each i, count how many j < i have x[j] == x[i]
// Equivalent to JAX's cum_count function
// x values must be in [0, numel)
//
// A single forward pass with a per-value running counter gives the identical
// result to the previous stable-sort + rank - n_less formulation: rank[i] minus
// the number of strictly-smaller elements is exactly i's position within its
// equal-value group in stable order, i.e. how many earlier j share x[i]. This is
// O(n) (was O(n log n) from a full stable_sort of n) and bit-identical (integer).
// `counts` is an optional caller-provided scratch buffer of size >= numel to
// avoid a per-call allocation; pass nullptr to allocate internally.
inline void cum_count(const int* x, int n, int numel, int* result,
                      int* counts = nullptr) {
    std::vector<int> owned;
    int* cnt;
    if (counts) {
        cnt = counts;
        std::fill(cnt, cnt + numel, 0);
    } else {
        owned.assign(numel, 0);
        cnt = owned.data();
    }
    for (int i = 0; i < n; i++) {
        int v = x[i];
        if (v >= 0 && v < numel) {
            result[i] = cnt[v]++;
        } else {
            result[i] = 0;
        }
    }
}

// random_assignment: assign n_products to n_workers round-robin after random permutation
// Returns product_worker_map[p] and product_ids_within_worker[p] for p in [0, n_products)
inline void random_assignment(
    uint64_t& rng_key,
    int n_products,
    int n_workers,
    int max_products_per_worker,
    int* product_worker_map,
    int* product_ids_within_worker
) {
    // Create permutation
    std::vector<int> perm(n_products);
    for (int i = 0; i < n_products; i++) perm[i] = i;
    // Fisher-Yates shuffle
    for (int i = n_products - 1; i > 0; --i) {
        rng_key = rng_key * 6364136223846793005ULL + 1442695040888963407ULL;
        int j = (int)((rng_key >> 33) % (i + 1));
        std::swap(perm[i], perm[j]);
    }

    // in_order_assigned_worker[k] = k % n_workers (for k in [0, n_workers*max_products_per_worker))
    // in_order_assigned_id[k] = k / n_workers
    // Then: product_worker_map[perm[i]] = in_order_assigned_worker[i]
    //       product_ids_within_worker[perm[i]] = in_order_assigned_id[i]
    for (int i = 0; i < n_products; i++) {
        int p = perm[i];
        product_worker_map[p] = i % n_workers;
        product_ids_within_worker[p] = i / n_workers;
    }
}

// compute_capacity_delta_other_threads on host
// fulfill: current fulfillment assignments [eff_num_steps]
// quantities: order quantities [eff_num_steps]
// order_2D_index: [n_workers, num_steps] indices into the eff_num_steps window (-1 for invalid)
// Output: capacity_delta [n_workers, num_steps, n_nodes_plus1]
inline void compute_capacity_delta_other_threads(
    int n_nodes_plus1,
    const int* fulfill,        // [eff_num_steps]
    const int* quantities,     // [eff_num_steps]
    const int* order_2D_index, // [n_workers * num_steps], values in [-1, eff_num_steps)
    int n_workers,
    int num_steps,
    int eff_num_steps,
    float* capacity_delta      // output: [n_workers * num_steps * n_nodes_plus1]
) {
    int total_2D = n_workers * num_steps;

    // Step 1: one_hot_fulfill[t, n] = quantity[t] if fulfill[t]==n, else 0
    // cumsum_fulfill[t, n] = -sum_{s<=t} one_hot_fulfill[s, n]
    // We compute this as running totals
    std::vector<float> cumsum(eff_num_steps * n_nodes_plus1, 0.0f);
    for (int t = 0; t < eff_num_steps; t++) {
        for (int n = 0; n < n_nodes_plus1; n++) {
            float prev = (t > 0) ? cumsum[(t - 1) * n_nodes_plus1 + n] : 0.0f;
            float delta = (fulfill[t] == n) ? -(float)quantities[t] : 0.0f;
            cumsum[t * n_nodes_plus1 + n] = prev + delta;
        }
    }

    // Step 2: For each entry in order_2D_index, gather from cumsum and compute diffs
    // caps[w][s] = cumsum[order_2D_index[w][s]]  (or 0 if index==-1)
    // cap_deltas[w][s] = caps[w][s] - caps[w][s-1]  (with caps[w][-1] = 0)
    // cap_deltas_by_product[w][s] = -one_hot_fulfill[order_2D_index[w][s]]
    // result = -(cap_deltas - cap_deltas_by_product)
    // The cumsum above is a serial prefix scan, but this gather is independent per
    // worker w, so parallelize the outer loop (each w writes a disjoint output
    // block; float ops per element are unchanged -> bit-identical).
    #pragma omp parallel for schedule(static)
    for (int w = 0; w < n_workers; w++) {
        for (int s = 0; s < num_steps; s++) {
            int idx = order_2D_index[w * num_steps + s];
            for (int n = 0; n < n_nodes_plus1; n++) {
                float cap_val = 0.0f;
                if (idx >= 0 && idx < eff_num_steps) {
                    cap_val = cumsum[idx * n_nodes_plus1 + n];
                }
                float cap_prev = 0.0f;
                if (s > 0) {
                    int prev_idx = order_2D_index[w * num_steps + s - 1];
                    if (prev_idx >= 0 && prev_idx < eff_num_steps) {
                        cap_prev = cumsum[prev_idx * n_nodes_plus1 + n];
                    }
                }
                float cap_delta = cap_val - cap_prev;

                float by_product = 0.0f;
                if (idx >= 0 && idx < eff_num_steps && fulfill[idx] == n) {
                    by_product = -(float)quantities[idx];
                }

                capacity_delta[(w * num_steps + s) * n_nodes_plus1 + n] =
                    -(cap_delta - by_product);
            }
        }
    }
}

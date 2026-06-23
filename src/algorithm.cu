#include "algorithm.h"
#include "types.h"
#include "utils.h"
#include "rng.h"
#include "simulation.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <chrono>

// Lightweight, env-gated host-side phase profiler (set PROFILE=1 to enable).
// The Picard bottleneck is host orchestration, so we time the CPU phases and the
// PCIe transfers separately to see where the per-iteration cost actually goes.
struct PhaseProf {
    double prep = 0, reshape = 0, h2d = 0, kernel = 0, d2h = 0, post = 0;
    long iters = 0;
    bool on = false;
};
static inline double prof_now() {
    return std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ============================================================
// Device allocation helpers
// ============================================================

AlgoState create_algo_state(
    const ProblemData& problem,
    const AlgoConfig& config,
    uint64_t seed
) {
    AlgoState state;
    state.iteration = 0;
    state.t_reset = 0;
    state.n_conflicts = 0;
    state.n_products = problem.n_products;
    state.n_nodes_plus1 = problem.n_nodes + 1;
    state.n_events = problem.n_events;
    state.rng_key = seed;

    int np1 = state.n_nodes_plus1;

    // Capacity: [n_nodes+1], with sentinel = inf
    std::vector<float> cap_host(np1);
    for (int i = 0; i < problem.n_nodes; i++) cap_host[i] = problem.capacity[i];
    cap_host[problem.n_nodes] = 1e30f;  // unfulfill sentinel capacity
    CUDA_CHECK(cudaMalloc(&state.capacity, np1 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(state.capacity, cap_host.data(), np1 * sizeof(float), cudaMemcpyHostToDevice));

    // Inventory: [n_products, n_nodes+1], with sentinel column = inf
    int inv_size = problem.n_products * np1;
    std::vector<float> inv_host(inv_size);
    for (int p = 0; p < problem.n_products; p++) {
        for (int n = 0; n < problem.n_nodes; n++) {
            inv_host[p * np1 + n] = problem.inventory[p * problem.n_nodes + n];
        }
        inv_host[p * np1 + problem.n_nodes] = 1e30f;  // sentinel
    }
    CUDA_CHECK(cudaMalloc(&state.inventory, inv_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(state.inventory, inv_host.data(), inv_size * sizeof(float), cudaMemcpyHostToDevice));

    // Fulfill: [n_events], initialized to sentinel (unfulfill)
    std::vector<int> fulfill_host(problem.n_events, problem.n_nodes);  // sentinel = n_nodes
    CUDA_CHECK(cudaMalloc(&state.fulfill, problem.n_events * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(state.fulfill, fulfill_host.data(), problem.n_events * sizeof(int), cudaMemcpyHostToDevice));

    return state;
}

void free_algo_state(AlgoState& state) {
    cudaFree(state.capacity);
    cudaFree(state.inventory);
    cudaFree(state.fulfill);
    state.capacity = nullptr;
    state.inventory = nullptr;
    state.fulfill = nullptr;
}

Event create_events(const ProblemData& problem) {
    Event ev;
    ev.n_events = problem.n_events;
    ev.n_nodes_plus1 = problem.n_nodes + 1;

    CUDA_CHECK(cudaMalloc(&ev.product, problem.n_events * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(ev.product, problem.order_products.data(),
                          problem.n_events * sizeof(int), cudaMemcpyHostToDevice));

    // Quantities are all 1 (as in JAX code)
    std::vector<int> ones(problem.n_events, 1);
    CUDA_CHECK(cudaMalloc(&ev.quantity, problem.n_events * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(ev.quantity, ones.data(),
                          problem.n_events * sizeof(int), cudaMemcpyHostToDevice));

    // node_index_near_to_far with unfulfill sentinel appended
    int np1 = problem.n_nodes + 1;
    std::vector<int> ntf_host(problem.n_events * np1);
    for (int i = 0; i < problem.n_events; i++) {
        for (int j = 0; j < problem.n_nodes; j++) {
            ntf_host[i * np1 + j] = problem.node_index_near_to_far[i * problem.n_nodes + j];
        }
        ntf_host[i * np1 + problem.n_nodes] = problem.n_nodes;  // sentinel
    }
    CUDA_CHECK(cudaMalloc(&ev.node_index_near_to_far, problem.n_events * np1 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(ev.node_index_near_to_far, ntf_host.data(),
                          problem.n_events * np1 * sizeof(int), cudaMemcpyHostToDevice));

    return ev;
}

void free_events(Event& ev) {
    cudaFree(ev.product);
    cudaFree(ev.quantity);
    cudaFree(ev.node_index_near_to_far);
    ev.product = nullptr;
    ev.quantity = nullptr;
    ev.node_index_near_to_far = nullptr;
}

NeuralNet create_neural_net(const AlgoConfig& config, uint64_t seed) {
    NeuralNet nn;
    nn.num_layers = (int)config.layer_sizes.size() - 1;
    nn.greedy_cost_prob = config.greedy_cost_prob;

    // Compute total params
    int total = 0;
    for (int i = 0; i < nn.num_layers; i++) {
        int nin = config.layer_sizes[i];
        int nout = config.layer_sizes[i + 1];
        total += nout * nin + nout;
    }
    nn.total_param_count = total;

    // Initialize params (random normal * 0.01)
    std::vector<float> params_host(total);
    uint64_t rng = seed;
    for (int i = 0; i < total; i++) {
        // Box-Muller for normal distribution
        float u1 = rand_uniform(rng);
        float u2 = rand_uniform(rng);
        float z = sqrtf(-2.0f * logf(u1 + 1e-30f)) * cosf(2.0f * 3.14159265f * u2);
        params_host[i] = z * 0.01f;
    }

    CUDA_CHECK(cudaMalloc(&nn.params, total * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(nn.params, params_host.data(), total * sizeof(float), cudaMemcpyHostToDevice));

    // Layer sizes on device
    CUDA_CHECK(cudaMalloc(&nn.layer_sizes, config.layer_sizes.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(nn.layer_sizes, config.layer_sizes.data(),
                          config.layer_sizes.size() * sizeof(int), cudaMemcpyHostToDevice));

    return nn;
}

void free_neural_net(NeuralNet& nn) {
    cudaFree(nn.params);
    cudaFree(nn.layer_sizes);
    nn.params = nullptr;
    nn.layer_sizes = nullptr;
}

void get_fulfill(const AlgoState& state, int* host_fulfill) {
    CUDA_CHECK(cudaMemcpy(host_fulfill, state.fulfill,
                          state.n_events * sizeof(int), cudaMemcpyDeviceToHost));
}

// ============================================================
// iterate_algorithm
// ============================================================
//
// The Picard loop advances only a small window (num_steps * n_workers events)
// per iteration, but the original implementation round-tripped the ENTIRE
// simulation state plus all (immutable) event arrays across PCIe on every call.
// IterContext keeps the authoritative mutable state resident on the host and the
// immutable event arrays cached, and reuses persistent device scratch buffers, so
// a single iteration only uploads what the kernel consumes and downloads its
// output — the per-iteration cost no longer scales with the full state size.

struct IterContext {
    int n_products = 0, np1 = 0, n_events = 0;
    int n_workers = 0, num_steps = 0, max_ppw = 0;
    size_t ev_count = 0;

    // Immutable event data — copied from the device once and reused. The full
    // near-to-far ordering is never needed on the host: it is immutable and already
    // device-resident (events.node_index_near_to_far), so the kernel indexes it
    // directly by global event index (P12) instead of a per-window host copy.
    std::vector<int> h_event_products;
    std::vector<int> h_event_quantities;

    // Authoritative mutable state, resident on the host across the whole loop.
    // (h_inventory is fully written by the init D2H before any read → no-init.)
    std::vector<float> h_capacity;
    NoInitVec<float> h_inventory;
    std::vector<int>   h_fulfill;

    // Persistent host scratch (sized once in init, refilled each iteration —
    // avoids ~15 heap allocations per Picard iteration and gives OpenMP stable
    // buffers to write into).
    // All persistent scratch below is fully overwritten before it is read each
    // iteration, so it uses the no-init allocator: resize() allocates without the
    // serial zero-fill, and the pages fault in (in parallel) during the OpenMP
    // write loops that fill them. This removed ~270 ms of init zero-fill (P11).
    int eff_num_steps = 0;
    NoInitVec<int> slice_products, slice_quantities, slice_fulfill;
    NoInitVec<int> product_worker_map, product_ids_within_worker;
    NoInitVec<int> workers, index_within_worker, cum_counts;
    NoInitVec<int> order_2D_index, valid_mask, order_2D_relative;
    // capacity_delta is now produced on the GPU (P9); only its small inputs
    // (slice_fulfill, slice_quantities, order_2D_relative) are uploaded.
    NoInitVec<int> ev2d_product, ev2d_quantity;
    NoInitVec<uint64_t> worker_keys;
    NoInitVec<int> fulfill_2D;
    NoInitVec<unsigned char> present;  // window presence bitmap for max_t_reset

    // Persistent device scratch (allocated once, reused each iteration).
    float* d_inv_scratch = nullptr;     // [n_products, np1] — fresh inventory copy/iter (P13)
    int* d_ev_product = nullptr;
    int* d_ev_quantity = nullptr;
    const int* d_event_ntf_full = nullptr;  // borrowed: events.node_index_near_to_far (not owned)
    int* d_order_2D_index = nullptr;        // global event idx per slot (replaces d_ev_ntf, P12)
    // GPU capacity-delta (P9): inputs uploaded, cumsum + cap_delta produced on device.
    int* d_slice_fulfill = nullptr;         // [eff]
    int* d_slice_quantities = nullptr;      // [eff]
    int* d_order_2D_relative = nullptr;     // [evc], in [0,eff) or -1
    float* d_cumsum = nullptr;              // [eff, np1] device scratch
    float* d_cap_delta = nullptr;           // [evc, np1] — now written by the gather kernel
    int* d_valid_mask = nullptr;
    uint64_t* d_worker_keys = nullptr;
    int* d_fulfill_2D = nullptr;

    PhaseProf prof;
};

// Stage state in from the device once, before the Picard loop.
static void init_iter_context(
    IterContext& ctx,
    const AlgoState& s,
    const Event& events,
    const AlgoConfig& config
) {
    ctx.n_products = s.n_products;
    ctx.np1 = s.n_nodes_plus1;
    ctx.n_events = s.n_events;
    ctx.n_workers = config.max_workers;
    ctx.num_steps = config.num_steps;
    ctx.max_ppw = config.max_products_per_worker;
    ctx.ev_count = (size_t)ctx.n_workers * ctx.num_steps;

    int np1 = ctx.np1;

    bool prof_init = (getenv("PROFILE") != nullptr);
    double ti = prof_init ? prof_now() : 0.0;
    auto mark = [&](const char* tag) {
        if (prof_init) { double n = prof_now(); printf("  [init] %-18s %7.1f ms\n", tag, 1000.0*(n-ti)); ti = n; }
    };

    // Cache immutable event scalars on the host (products/quantities). The full
    // near-to-far ordering stays only on the device (borrowed pointer); the kernel
    // indexes it by global event index, so no host copy / D2H of it is needed (P12).
    ctx.h_event_products.resize(ctx.n_events);
    ctx.h_event_quantities.resize(ctx.n_events);
    CUDA_CHECK(cudaMemcpy(ctx.h_event_products.data(), events.product,
                          ctx.n_events * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ctx.h_event_quantities.data(), events.quantity,
                          ctx.n_events * sizeof(int), cudaMemcpyDeviceToHost));
    ctx.d_event_ntf_full = events.node_index_near_to_far;
    mark("D2H events");

    // Pull initial mutable state once; the host copy is authoritative thereafter.
    ctx.h_capacity.resize(np1);
    ctx.h_inventory.resize((size_t)ctx.n_products * np1);
    ctx.h_fulfill.resize(ctx.n_events);
    CUDA_CHECK(cudaMemcpy(ctx.h_capacity.data(), s.capacity,
                          np1 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ctx.h_inventory.data(), s.inventory,
                          (size_t)ctx.n_products * np1 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ctx.h_fulfill.data(), s.fulfill,
                          ctx.n_events * sizeof(int), cudaMemcpyDeviceToHost));
    mark("D2H state");

    ctx.prof.on = (getenv("PROFILE") != nullptr);

    // Size persistent prep scratch once. eff_num_steps and the window dims depend
    // only on config + n_events, so they are fixed across the whole loop.
    ctx.eff_num_steps = std::min(ctx.num_steps * ctx.n_workers, ctx.n_events);
    int ens = ctx.eff_num_steps;
    size_t evc = ctx.ev_count;
    ctx.slice_products.resize(ens);
    ctx.slice_quantities.resize(ens);
    ctx.slice_fulfill.resize(ens);
    ctx.product_worker_map.resize(ctx.n_products);
    ctx.product_ids_within_worker.resize(ctx.n_products);
    ctx.workers.resize(ens);
    ctx.index_within_worker.resize(ens);
    ctx.cum_counts.resize(ctx.n_workers);
    ctx.order_2D_index.resize(evc);
    ctx.valid_mask.resize(evc);
    ctx.order_2D_relative.resize(evc);
    ctx.ev2d_product.resize(evc);
    ctx.ev2d_quantity.resize(evc);
    ctx.worker_keys.resize(ctx.n_workers);
    ctx.fulfill_2D.resize(evc);
    ctx.present.resize(ens);
    mark("host resizes");

    // Per-iteration buffer sizes depend only on config dims, so allocate once.
    // d_inv_scratch holds the full inventory ([n_products, np1]) — a fresh copy of
    // the authoritative host inventory is uploaded into it each iteration (P13).
    CUDA_CHECK(cudaMalloc(&ctx.d_inv_scratch, (size_t)ctx.n_products * np1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx.d_ev_product, ctx.ev_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.d_ev_quantity, ctx.ev_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.d_order_2D_index, ctx.ev_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.d_cap_delta, ctx.ev_count * np1 * sizeof(float)));
    // GPU capacity-delta scratch + inputs (P9).
    CUDA_CHECK(cudaMalloc(&ctx.d_slice_fulfill, (size_t)ens * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.d_slice_quantities, (size_t)ens * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.d_order_2D_relative, ctx.ev_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.d_cumsum, (size_t)ens * np1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx.d_valid_mask, ctx.ev_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.d_worker_keys, ctx.n_workers * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&ctx.d_fulfill_2D, ctx.ev_count * sizeof(int)));
    mark("cudaMalloc");
}

// Push the final authoritative host state back to the device once, after the loop.
static void finalize_iter_context(IterContext& ctx, AlgoState& s) {
    CUDA_CHECK(cudaMemcpy(s.capacity, ctx.h_capacity.data(),
                          ctx.np1 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.inventory, ctx.h_inventory.data(),
                          (size_t)ctx.n_products * ctx.np1 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.fulfill, ctx.h_fulfill.data(),
                          ctx.n_events * sizeof(int), cudaMemcpyHostToDevice));
}

static void free_iter_context(IterContext& ctx) {
    cudaFree(ctx.d_inv_scratch);
    cudaFree(ctx.d_ev_product);
    cudaFree(ctx.d_ev_quantity);
    cudaFree(ctx.d_order_2D_index);
    cudaFree(ctx.d_cap_delta);
    cudaFree(ctx.d_slice_fulfill);
    cudaFree(ctx.d_slice_quantities);
    cudaFree(ctx.d_order_2D_relative);
    cudaFree(ctx.d_cumsum);
    cudaFree(ctx.d_valid_mask);
    cudaFree(ctx.d_worker_keys);
    cudaFree(ctx.d_fulfill_2D);
}

// One Picard iteration, operating entirely on the host-resident context state.
// Only the kernel inputs are uploaded and only fulfill_2D is read back.
static void iterate_core(
    AlgoState& algo_state,
    const NeuralNet& nn,
    const AlgoConfig& config,
    IterContext& ctx
) {
    int n_products = ctx.n_products;
    int np1 = ctx.np1;
    int n_events = ctx.n_events;
    int n_workers = ctx.n_workers;
    int num_steps = ctx.num_steps;
    int max_ppw = ctx.max_ppw;

    // Authoritative state lives in the context — no per-iteration D2H copies.
    std::vector<float>& h_capacity = ctx.h_capacity;
    auto& h_inventory = ctx.h_inventory;
    std::vector<int>& h_fulfill = ctx.h_fulfill;
    std::vector<int>& h_event_products = ctx.h_event_products;
    std::vector<int>& h_event_quantities = ctx.h_event_quantities;

    PhaseProf& prof = ctx.prof;
    double tp = prof.on ? prof_now() : 0.0;

    // Bind persistent scratch (allocated once in init_iter_context).
    auto& slice_products = ctx.slice_products;
    auto& slice_quantities = ctx.slice_quantities;
    auto& slice_fulfill = ctx.slice_fulfill;
    auto& product_worker_map = ctx.product_worker_map;
    auto& product_ids_within_worker = ctx.product_ids_within_worker;
    auto& workers = ctx.workers;
    auto& index_within_worker = ctx.index_within_worker;
    auto& order_2D_index = ctx.order_2D_index;
    auto& valid_mask = ctx.valid_mask;
    auto& order_2D_relative = ctx.order_2D_relative;
    auto& ev2d_product = ctx.ev2d_product;
    auto& ev2d_quantity = ctx.ev2d_quantity;

    // --- Step 1: Build 2D event structure ---
    int eff_num_steps = ctx.eff_num_steps;
    int t0 = std::min(algo_state.t_reset, n_events - eff_num_steps);

    // get_slice: extract [t0, t0+eff_num_steps) and roll by (t_reset - t0). The
    // roll is a rotation, so emit it as two contiguous copies instead of a
    // per-element double modulo (P7): out[0..eff-k) = arr[t0+k..t0+eff),
    // out[eff-k..eff) = arr[t0..t0+k), with k = (t_reset - t0) mod eff.
    int k = (algo_state.t_reset - t0) % eff_num_steps;
    auto get_slice_int = [&](const auto& arr, auto& out) {
        std::copy(arr.begin() + t0 + k, arr.begin() + t0 + eff_num_steps, out.begin());
        std::copy(arr.begin() + t0, arr.begin() + t0 + k, out.begin() + (eff_num_steps - k));
    };
    get_slice_int(h_event_products, slice_products);
    get_slice_int(h_event_quantities, slice_quantities);
    get_slice_int(h_fulfill, slice_fulfill);

    // Random assignment of products to workers (Fisher-Yates is inherently serial)
    uint64_t key = algo_state.rng_key;
    uint64_t subkey;
    split_key(key, key, subkey);

    random_assignment(subkey, n_products, n_workers, max_ppw,
                      product_worker_map.data(), product_ids_within_worker.data());

    // Map events to workers
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < eff_num_steps; i++) {
        workers[i] = product_worker_map[slice_products[i]];
    }

    // cum_count within workers (O(n) running counter; serial dependency, uses
    // persistent scratch). Fast enough that parallelizing is unnecessary.
    cum_count(workers.data(), eff_num_steps, n_workers, index_within_worker.data(),
              ctx.cum_counts.data());

    // Build order_2D_index: [n_workers, num_steps]. cum_count guarantees a unique
    // step s per worker, so every (w,s) target is written at most once -> the
    // scatter is race-free and parallelizable. Reset to -1 first (persistent buf).
    std::fill(order_2D_index.begin(), order_2D_index.end(), -1);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < eff_num_steps; i++) {
        int w = workers[i];
        int s = index_within_worker[i];
        if (s < num_steps) {
            order_2D_index[w * num_steps + s] = algo_state.t_reset + i;
        }
    }

    // Valid mask + relative indices (independent per element)
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_workers * num_steps; i++) {
        int v = (order_2D_index[i] >= 0 && order_2D_index[i] < n_events) ? 1 : 0;
        valid_mask[i] = v;
        order_2D_relative[i] = v ? (order_2D_index[i] - algo_state.t_reset) : -1;
    }

    // capacity_delta is computed on the GPU now (P9) — see Step 3, after its inputs
    // (slice_fulfill, slice_quantities, order_2D_relative) are uploaded.

    // --- Step 2: Build the per-slot event scalars for the kernel ---
    // No inventory reshape anymore: the kernel reads each event's inventory row
    // straight from the full device-resident inventory (d_inv_scratch), indexed by
    // the original product id — products are partitioned 1:1 to workers, so the rows
    // a worker touches are disjoint from every other worker's (P13). The near-to-far
    // ordering is likewise gathered on-device by global index (P12). So all that is
    // left to stage per slot is the product id and quantity.
    double tr = prof.on ? prof_now() : 0.0;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_workers * num_steps; i++) {
        int idx = order_2D_index[i];
        if (valid_mask[i]) {
            ev2d_product[i] = h_event_products[idx];   // original product id (P13)
            ev2d_quantity[i] = h_event_quantities[idx];
        } else {
            ev2d_product[i] = 0;
            ev2d_quantity[i] = 0;
        }
    }

    if (prof.on) prof.reshape += prof_now() - tr;

    // Prepare RNG keys for workers (split_key chains serially)
    auto& worker_keys = ctx.worker_keys;
    uint64_t wk = algo_state.rng_key;
    for (int w = 0; w < n_workers; w++) {
        split_key(wk, wk, worker_keys[w]);
    }

    // --- Step 3: Upload kernel inputs and launch (reusing persistent buffers) ---
    size_t ev_count = ctx.ev_count;

    if (prof.on) { prof.prep += prof_now() - tp; tp = prof_now(); }

    // Upload a fresh copy of the authoritative inventory (70 MB) for the kernel to
    // mutate as per-iteration scratch (P13) — replaces the 124 MB worker_inv reshape.
    CUDA_CHECK(cudaMemcpy(ctx.d_inv_scratch, h_inventory.data(),
                          (size_t)n_products * np1 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx.d_ev_product, ev2d_product.data(), ev_count * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx.d_ev_quantity, ev2d_quantity.data(), ev_count * sizeof(int), cudaMemcpyHostToDevice));
    // P12: upload the per-slot global event index (4 MB) instead of the per-window
    // ntf (124 MB); the kernel gathers the ntf row from the device-resident full ntf.
    CUDA_CHECK(cudaMemcpy(ctx.d_order_2D_index, order_2D_index.data(), ev_count * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx.d_valid_mask, valid_mask.data(), ev_count * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx.d_worker_keys, worker_keys.data(), n_workers * sizeof(uint64_t), cudaMemcpyHostToDevice));
    // P9: upload only capacity_delta's small inputs (3×4 MB) and produce the 124 MB
    // capacity_delta on the device — no host cumsum/gather, no 124 MB H2D.
    CUDA_CHECK(cudaMemcpy(ctx.d_slice_fulfill, slice_fulfill.data(), eff_num_steps * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx.d_slice_quantities, slice_quantities.data(), eff_num_steps * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx.d_order_2D_relative, order_2D_relative.data(), ev_count * sizeof(int), cudaMemcpyHostToDevice));

    // The kernel reads its initial capacity from the device; sync the current
    // (pre-step) host capacity into the device buffer just before launch.
    CUDA_CHECK(cudaMemcpy(algo_state.capacity, h_capacity.data(),
                          np1 * sizeof(float), cudaMemcpyHostToDevice));

    if (prof.on) { prof.h2d += prof_now() - tp; tp = prof_now(); }

    // P9: produce capacity_delta on the device. The per-column cumsum (serial in t,
    // exact since quantities are integers) feeds a fully-parallel gather. Same stream
    // as the fulfillment kernel below, so it is ordered before the kernel reads it.
    capacity_cumsum_kernel<<<1, np1>>>(
        ctx.d_slice_fulfill, ctx.d_slice_quantities, eff_num_steps, np1, ctx.d_cumsum);
    {
        long total = (long)n_workers * num_steps * np1;
        int threads = 256;
        int blocks = (int)((total + threads - 1) / threads);
        capacity_gather_kernel<<<blocks, threads>>>(
            ctx.d_cumsum, ctx.d_order_2D_relative, ctx.d_slice_fulfill, ctx.d_slice_quantities,
            n_workers, num_steps, eff_num_steps, np1, ctx.d_cap_delta);
    }

    // Launch the fulfillment kernel. Default: warp-cooperative (P8) — one warp per
    // worker, parallelizing the per-event MLP across lanes. Set LEGACY_KERNEL=1 to
    // use the original one-thread-per-worker scan (bit-identical reference).
    static const bool legacy_kernel = (getenv("LEGACY_KERNEL") != nullptr);
    if (legacy_kernel) {
        simulate_worker_kernel<<<n_workers, 1>>>(
            nn.params, nn.layer_sizes, nn.num_layers, nn.greedy_cost_prob,
            ctx.d_inv_scratch, algo_state.capacity,
            ctx.d_ev_product, ctx.d_ev_quantity,
            ctx.d_event_ntf_full, ctx.d_order_2D_index, ctx.d_cap_delta,
            ctx.d_valid_mask, ctx.d_worker_keys,
            n_workers, num_steps, max_ppw, np1,
            ctx.d_fulfill_2D
        );
    } else {
        int blocks = (n_workers + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        simulate_worker_kernel_warp<<<blocks, WARPS_PER_BLOCK * 32>>>(
            nn.params, nn.layer_sizes, nn.num_layers, nn.greedy_cost_prob,
            ctx.d_inv_scratch, algo_state.capacity,
            ctx.d_ev_product, ctx.d_ev_quantity,
            ctx.d_event_ntf_full, ctx.d_order_2D_index, ctx.d_cap_delta,
            ctx.d_valid_mask, ctx.d_worker_keys,
            n_workers, num_steps, max_ppw, np1,
            ctx.d_fulfill_2D
        );
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (prof.on) { prof.kernel += prof_now() - tp; tp = prof_now(); }

    // Read back fulfill_2D
    auto& fulfill_2D = ctx.fulfill_2D;
    CUDA_CHECK(cudaMemcpy(fulfill_2D.data(), ctx.d_fulfill_2D, ev_count * sizeof(int), cudaMemcpyDeviceToHost));

    if (prof.on) { prof.d2h += prof_now() - tp; tp = prof_now(); }

    // --- Step 4: max_t_reset = end of the contiguous run of assigned global
    //     indices starting at t_reset (P-post). Each assigned 2D slot holds a
    //     distinct global index t_reset+i with i in [0, eff_num_steps); event 0
    //     always lands at t_reset, so the run starts there and max_t_reset is just
    //     t_reset + (run length). Mark presence over the window and find the first
    //     hole — O(window) — replacing the previous O(window log window) sort.
    //     This reproduces the sort-based value exactly (the first gap among the
    //     sorted assigned indices is the first missing offset above the run).
    auto& present = ctx.present;
    std::fill(present.begin(), present.begin() + eff_num_steps, (unsigned char)0);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_workers * num_steps; i++) {
        int g = order_2D_index[i];
        if (g >= 0) present[g - algo_state.t_reset] = 1;  // offset in [0, eff)
    }
    int run = 0;
    while (run < eff_num_steps && present[run]) run++;
    int max_t_reset = algo_state.t_reset + run;

    // --- Step 5: Conflict detection (window-scoped) ---
    // The new fulfillment differs from the committed one only at assigned window
    // indices (all >= t_reset), so the first conflict is the lowest assigned index
    // whose new value differs from the old. Scan the window instead of all
    // n_events. Read the old value here, before Step 6 overwrites it in place.
    int first_conflict = max_t_reset;
    for (int i = 0; i < n_workers * num_steps; i++) {
        if (!valid_mask[i]) continue;
        int idx = order_2D_index[i];
        if (fulfill_2D[i] != h_fulfill[idx] && idx < first_conflict) {
            first_conflict = idx;
        }
    }
    int new_t_reset = std::min(first_conflict, max_t_reset);

    // --- Step 6: Commit window fulfillments in place, then update inventory and
    //            capacity over the newly resolved range [t_reset, new_t_reset). ---
    for (int i = 0; i < n_workers * num_steps; i++) {
        if (valid_mask[i]) {
            h_fulfill[order_2D_index[i]] = fulfill_2D[i];
        }
    }
    int update_end = std::min(new_t_reset, n_events);
    for (int i = algo_state.t_reset; i < update_end; i++) {
        float qty = (float)h_event_quantities[i];
        int prod = h_event_products[i];
        int node = h_fulfill[i];
        h_inventory[prod * np1 + node] -= qty;
        h_capacity[node] -= qty;
    }

    bool is_conflict = (new_t_reset < max_t_reset) && (new_t_reset != algo_state.t_reset);

    // State stays resident on the host (h_fulfill updated in place above);
    // finalize_iter_context flushes the final state to the device after the loop.
    algo_state.iteration++;
    algo_state.t_reset = new_t_reset;
    algo_state.n_conflicts += is_conflict ? 1 : 0;
    algo_state.rng_key = key;

    if (prof.on) { prof.post += prof_now() - tp; prof.iters++; }
}

static void report_profile(const PhaseProf& p) {
    if (!p.on || p.iters == 0) return;
    double total = p.prep + p.h2d + p.kernel + p.d2h + p.post;
    if (total <= 0) return;
    auto line = [&](const char* name, double t) {
        printf("  %-16s %9.4f s  %5.1f%%  %9.3f ms/iter\n",
               name, t, 100.0 * t / total, 1000.0 * t / p.iters);
    };
    printf("=== Picard host profile (%ld iterations) ===\n", p.iters);
    line("prep (CPU)",   p.prep);
    line("  of which:reshape", p.reshape);
    line("H2D upload",   p.h2d);
    line("kernel",       p.kernel);
    line("D2H readback", p.d2h);
    line("post (CPU)",   p.post);
    printf("  %-16s %9.4f s\n", "measured total", total);
}

// Public single-iteration entry point: stage state in/out around one core step
// so it remains self-contained (device state is consistent on entry and exit).
void iterate_algorithm(
    AlgoState& algo_state,
    const Event& events,
    const NeuralNet& nn,
    const AlgoConfig& config
) {
    IterContext ctx;
    init_iter_context(ctx, algo_state, events, config);
    iterate_core(algo_state, nn, config, ctx);
    finalize_iter_context(ctx, algo_state);
    free_iter_context(ctx);
}

// ============================================================
// run_algorithm_day
// ============================================================

void run_algorithm_day(
    AlgoState& algo_state,
    const Event& events,
    const NeuralNet& nn,
    const AlgoConfig& config
) {
    // Stage state in once, iterate without crossing the PCIe boundary for the
    // full state, then flush once.
    IterContext ctx;
    init_iter_context(ctx, algo_state, events, config);
    while (algo_state.iteration < config.max_iters &&
           algo_state.t_reset < algo_state.n_events) {
        iterate_core(algo_state, nn, config, ctx);
    }
    finalize_iter_context(ctx, algo_state);
    report_profile(ctx.prof);
    free_iter_context(ctx);
}

// ============================================================
// Sequential simulation (no Picard iteration)
// ============================================================
// A single-threaded CUDA kernel that processes ALL events in order.
// No worker partitioning, no conflict detection — just a straight scan.

__global__ void simulate_sequential_kernel(
    const float* nn_params,
    const int* nn_layer_sizes,
    int nn_num_layers,
    float greedy_cost_prob,
    float* inventory,              // [n_products, n_nodes_plus1] — modified in place
    float* capacity,               // [n_nodes_plus1] — modified in place
    const int* event_products,     // [n_events]
    const int* event_quantities,   // [n_events]
    const int* event_ntf,          // [n_events, n_nodes_plus1]
    int n_events,
    int n_nodes_plus1,
    uint64_t rng_key,
    int* fulfill                   // [n_events] output
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    float scratch_a[MAX_NN_DIM];
    float scratch_b[MAX_NN_DIM];
    float nn_output[MAX_NN_DIM];

    uint64_t key = rng_key;

    for (int i = 0; i < n_events; i++) {
        int product = event_products[i];
        int quantity = event_quantities[i];
        const int* node_ntf = event_ntf + (size_t)i * n_nodes_plus1;
        float* inv_row = inventory + (size_t)product * n_nodes_plus1;

        uint64_t subkey;
        split_key(key, key, subkey);

        int node = fulfillment(
            subkey, greedy_cost_prob,
            nn_params, nn_layer_sizes, nn_num_layers,
            inv_row, capacity, quantity,
            node_ntf, n_nodes_plus1,
            scratch_a, scratch_b, nn_output);

        fulfill[i] = node;

        // Update state
        inv_row[node] -= (float)quantity;
        capacity[node] -= (float)quantity;
    }
}

void run_sequential_day(
    AlgoState& algo_state,
    const Event& events,
    const NeuralNet& nn,
    const AlgoConfig& config
) {
    // Launch a single-thread kernel that processes all events sequentially
    simulate_sequential_kernel<<<1, 1>>>(
        nn.params, nn.layer_sizes, nn.num_layers, nn.greedy_cost_prob,
        algo_state.inventory,
        algo_state.capacity,
        events.product,
        events.quantity,
        events.node_index_near_to_far,
        algo_state.n_events,
        algo_state.n_nodes_plus1,
        algo_state.rng_key,
        algo_state.fulfill
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    algo_state.t_reset = algo_state.n_events;
}

// ============================================================
// Benchmark kernels
// ============================================================

// 1. Dynamics only: greedy nearest feasible node, no NN.
//    Measures pure state-update throughput.
__global__ void bench_dynamics_only_kernel(
    float* inventory,              // [n_products, n_nodes_plus1]
    float* capacity,               // [n_nodes_plus1]
    const int* event_products,     // [n_steps]
    const int* event_quantities,   // [n_steps]
    const int* event_ntf,          // [n_steps, n_nodes_plus1]
    int n_steps,
    int n_nodes_plus1,
    int* fulfill
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    for (int i = 0; i < n_steps; i++) {
        int product = event_products[i];
        int quantity = event_quantities[i];
        const int* node_ntf = event_ntf + (size_t)i * n_nodes_plus1;
        float* inv_row = inventory + (size_t)product * n_nodes_plus1;

        // Greedy nearest: first feasible node in near-to-far order
        int node = node_ntf[n_nodes_plus1 - 1]; // default: unfulfill
        for (int rank = 0; rank < n_nodes_plus1; rank++) {
            int n = node_ntf[rank];
            if (capacity[n] >= (float)quantity && inv_row[n] >= (float)quantity) {
                node = n;
                break;
            }
        }

        fulfill[i] = node;
        inv_row[node] -= (float)quantity;
        capacity[node] -= (float)quantity;
    }
}

// 1b. Dynamics only, OPTIMIZED.
//   - capacity cached in shared memory (read once, mutate on-chip, write back once)
//   - __restrict__ pointers so the compiler can cache global reads
//   - cross-iteration L2 prefetch of the next event's inventory row to hide the
//     dominant random global-load latency in this single-thread sequential scan
__global__ void bench_dynamics_only_kernel_opt(
    float* __restrict__ inventory,        // [n_products, n_nodes_plus1]
    float* __restrict__ capacity,         // [n_nodes_plus1]
    const int* __restrict__ event_products,     // [n_steps]
    const int* __restrict__ event_quantities,   // [n_steps]
    const int* __restrict__ event_ntf,          // [n_steps, n_nodes_plus1]
    int n_steps,
    int n_nodes_plus1,
    int* __restrict__ fulfill
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const int np1 = n_nodes_plus1;
    const int PF_DIST = 8;  // software-pipeline depth: prefetch this many events ahead

    // Hot capacity vector lives on-chip for the whole scan.
    __shared__ float cap[MAX_NODES];
    for (int n = 0; n < np1; n++) cap[n] = capacity[n];

    // Warm the pipeline: kick off the first PF_DIST inventory-row loads.
    for (int j = 0; j < PF_DIST && j < n_steps; j++) {
        const float* p = inventory + (size_t)event_products[j] * np1;
        asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
    }

    for (int i = 0; i < n_steps; i++) {
        int product = event_products[i];
        float q = (float)event_quantities[i];
        const int* __restrict__ node_ntf = event_ntf + (size_t)i * np1;
        float* __restrict__ inv_row = inventory + (size_t)product * np1;

        // Keep the pipeline full: prefetch the event PF_DIST ahead.
        if (i + PF_DIST < n_steps) {
            const float* nxt = inventory + (size_t)event_products[i + PF_DIST] * np1;
            asm volatile("prefetch.global.L2 [%0];" :: "l"(nxt));
        }

        // Greedy nearest: first feasible node in near-to-far order
        int node = node_ntf[np1 - 1];  // default: unfulfill sentinel
        #pragma unroll 4
        for (int rank = 0; rank < np1; rank++) {
            int n = node_ntf[rank];
            if (cap[n] >= q && inv_row[n] >= q) { node = n; break; }
        }

        fulfill[i] = node;
        inv_row[node] -= q;
        cap[node] -= q;
    }

    // Persist capacity back to global (parity with the in-place baseline).
    for (int n = 0; n < np1; n++) capacity[n] = cap[n];
}

// 1c. Dynamics only, WARP-PARALLEL (requires n_nodes_plus1 <= 32).
//   One warp processes events sequentially, but each event's near-to-far
//   feasibility scan is done by all 32 lanes at once: lane r checks rank r,
//   the whole inventory row loads as one coalesced transaction, and __ballot +
//   __ffs selects the first feasible node in near-to-far order. The cross-event
//   capacity recurrence stays sequential in shared memory.
__global__ void bench_dynamics_only_kernel_warp(
    float* __restrict__ inventory,        // [n_products, n_nodes_plus1]
    float* __restrict__ capacity,         // [n_nodes_plus1]
    const int* __restrict__ event_products,
    const int* __restrict__ event_quantities,
    const int* __restrict__ event_ntf,
    int n_steps,
    int n_nodes_plus1,
    int* __restrict__ fulfill
) {
    if (blockIdx.x != 0) return;
    const int lane = threadIdx.x;
    if (lane >= 32) return;
    const int np1 = n_nodes_plus1;        // assumed <= 32

    __shared__ float cap[MAX_NODES];
    if (lane < np1) cap[lane] = capacity[lane];
    __syncwarp();

    for (int i = 0; i < n_steps; i++) {
        int product = event_products[i];
        float q = (float)event_quantities[i];
        const int* __restrict__ node_ntf = event_ntf + (size_t)i * np1;
        const float* __restrict__ inv_row = inventory + (size_t)product * np1;

        // Each lane evaluates one rank in near-to-far order.
        int node_at_rank = -1;
        int feasible = 0;
        if (lane < np1) {
            node_at_rank = node_ntf[lane];
            feasible = (cap[node_at_rank] >= q && inv_row[node_at_rank] >= q) ? 1 : 0;
        }
        unsigned ballot = __ballot_sync(0xffffffffu, feasible);

        // The owner lane (lowest feasible rank) already holds the chosen node in a
        // register, so it does the update directly — no __shfl, and the unfulfill
        // sentinel row is only read when nothing is feasible (off the common path).
        if (ballot == 0u) {
            if (lane == 0) fulfill[i] = node_ntf[np1 - 1];             // unfulfill sentinel
        } else {
            int first = __ffs((int)ballot) - 1;                        // lowest feasible rank
            if (lane == first) {
                fulfill[i] = node_at_rank;
                inventory[(size_t)product * np1 + node_at_rank] -= q;  // global RMW
                cap[node_at_rank] -= q;                                // shared
            }
        }
        __threadfence_block();   // make the inventory store visible to later events
        __syncwarp();            // publish the shared capacity update to all lanes (single warp)
    }

    if (lane < np1) capacity[lane] = cap[lane];
}

// 2. NN only: run NN inference per event, write output norm to fulfill (as dummy).
//    No state update — measures pure NN throughput.
__global__ void bench_nn_only_kernel(
    const float* nn_params,
    const int* nn_layer_sizes,
    int nn_num_layers,
    const float* inventory,        // [n_products, n_nodes_plus1] — read only
    const int* event_products,     // [n_steps]
    int n_steps,
    int n_nodes_plus1,
    int* fulfill                   // dummy output
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    float scratch_a[MAX_NN_DIM];
    float scratch_b[MAX_NN_DIM];
    float nn_output[MAX_NN_DIM];

    for (int i = 0; i < n_steps; i++) {
        int product = event_products[i];
        const float* inv_row = inventory + (size_t)product * n_nodes_plus1;

        nn_predict(nn_params, nn_layer_sizes, nn_num_layers,
                   inv_row, nn_output, scratch_a, scratch_b);

        // Write something to prevent the compiler from optimizing away the NN
        float norm = 0.0f;
        for (int j = 0; j < n_nodes_plus1; j++) norm += nn_output[j] * nn_output[j];
        fulfill[i] = (int)(norm * 1e6f);
    }
}

// 3. Full: NN + fulfillment + state update (same as sequential kernel)
//    Already exists as simulate_sequential_kernel, so we reuse it.

void run_benchmark(
    const AlgoState& algo_state,
    const Event& events,
    const NeuralNet& nn,
    int n_steps
) {
    int np1 = algo_state.n_nodes_plus1;
    int n_products = algo_state.n_products;
    int n_events = algo_state.n_events;

    if (n_steps > n_events) n_steps = n_events;

    printf("=== Benchmark: %d events, %d nodes, %d products ===\n",
           n_steps, np1 - 1, n_products);

    // We need fresh copies of inventory/capacity for each run since the kernels mutate them
    size_t inv_bytes = (size_t)n_products * np1 * sizeof(float);
    size_t cap_bytes = (size_t)np1 * sizeof(float);

    float *d_inv_copy, *d_cap_copy;
    int* d_fulfill_out;
    CUDA_CHECK(cudaMalloc(&d_inv_copy, inv_bytes));
    CUDA_CHECK(cudaMalloc(&d_cap_copy, cap_bytes));
    CUDA_CHECK(cudaMalloc(&d_fulfill_out, n_steps * sizeof(int)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    auto run_timed = [&](const char* label, auto launch_fn) {
        // Reset state
        CUDA_CHECK(cudaMemcpy(d_inv_copy, algo_state.inventory, inv_bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_cap_copy, algo_state.capacity, cap_bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemset(d_fulfill_out, 0, n_steps * sizeof(int)));
        CUDA_CHECK(cudaDeviceSynchronize());

        // Warmup
        launch_fn(d_inv_copy, d_cap_copy, d_fulfill_out);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Reset state again after warmup
        CUDA_CHECK(cudaMemcpy(d_inv_copy, algo_state.inventory, inv_bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_cap_copy, algo_state.capacity, cap_bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());

        // Timed run
        CUDA_CHECK(cudaEventRecord(start));
        launch_fn(d_inv_copy, d_cap_copy, d_fulfill_out);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        double us_per_event = (double)ms * 1000.0 / n_steps;
        printf("  %-20s %8.3f ms  (%6.3f us/event)\n", label, ms, us_per_event);
    };

    // 1. Dynamics only (baseline)
    run_timed("dynamics-only", [&](float* inv, float* cap, int* ful) {
        bench_dynamics_only_kernel<<<1, 1>>>(
            inv, cap,
            events.product, events.quantity, events.node_index_near_to_far,
            n_steps, np1, ful);
    });

    // Capture baseline fulfillment for a correctness check vs the optimized kernel.
    std::vector<int> baseline_fulfill(n_steps);
    CUDA_CHECK(cudaMemcpy(baseline_fulfill.data(), d_fulfill_out,
                          n_steps * sizeof(int), cudaMemcpyDeviceToHost));

    // 1b. Dynamics only (optimized)
    run_timed("dynamics-only-opt", [&](float* inv, float* cap, int* ful) {
        bench_dynamics_only_kernel_opt<<<1, 1>>>(
            inv, cap,
            events.product, events.quantity, events.node_index_near_to_far,
            n_steps, np1, ful);
    });

    // Verify the optimized kernel produces identical fulfillment decisions.
    auto check_against_baseline = [&](const char* label) {
        std::vector<int> got(n_steps);
        CUDA_CHECK(cudaMemcpy(got.data(), d_fulfill_out,
                              n_steps * sizeof(int), cudaMemcpyDeviceToHost));
        int mism = 0;
        for (int i = 0; i < n_steps; i++)
            if (got[i] != baseline_fulfill[i]) mism++;
        printf("  %-20s %s (%d/%d mismatches)\n", label,
               mism == 0 ? "OK — identical" : "MISMATCH", mism, n_steps);
    };
    check_against_baseline("dynamics-opt check");

    // 1c. Dynamics only (warp-parallel) — only valid when a node row fits in a warp.
    if (np1 <= 32) {
        run_timed("dynamics-only-warp", [&](float* inv, float* cap, int* ful) {
            bench_dynamics_only_kernel_warp<<<1, 32>>>(
                inv, cap,
                events.product, events.quantity, events.node_index_near_to_far,
                n_steps, np1, ful);
        });
        check_against_baseline("dynamics-warp check");
    }

    // 2. NN only
    run_timed("nn-only", [&](float* inv, float* cap, int* ful) {
        (void)cap; // unused
        bench_nn_only_kernel<<<1, 1>>>(
            nn.params, nn.layer_sizes, nn.num_layers,
            inv,  // read-only, but we pass the copy for consistency
            events.product,
            n_steps, np1, ful);
    });

    // 3. Full (NN + fulfillment + dynamics)
    run_timed("full", [&](float* inv, float* cap, int* ful) {
        simulate_sequential_kernel<<<1, 1>>>(
            nn.params, nn.layer_sizes, nn.num_layers, nn.greedy_cost_prob,
            inv, cap,
            events.product, events.quantity, events.node_index_near_to_far,
            n_steps, np1, algo_state.rng_key, ful);
    });

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    cudaFree(d_inv_copy);
    cudaFree(d_cap_copy);
    cudaFree(d_fulfill_out);
}

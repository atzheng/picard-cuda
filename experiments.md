# Experiments log

Benchmark runs of the fulfillment simulator: sequential single-thread scan vs.
parallel Picard iteration on GPU. Each entry records the commit the binary was
built from, the workload, the configuration, measured timings, and a correctness
check (parallel must be bit-identical to sequential on the same data).

**Hardware:** NVIDIA A100 80GB PCIe (driver 595.71.05), 24-core host CPU.

---

## E1 — Sequential vs. parallel at 1,000,000 events

- **Commit:** `54dcf3e` (Initial commit: optimized Picard host loop)
- **Workload:** `data/sub1m/npy` — first 1,000,000 events of the full dataset,
  product ids remapped to a compact range. **567,323 distinct products, 30 nodes.**
  Generated from `data/npy` (1M products × 30 nodes × 3M events).
- **Seed:** 42 (deterministic iteration count / conflicts).
- **Correctness oracle:** parallel fulfillment vs. sequential fulfillment,
  compared with `np.array_equal`.

| Mode | Configuration | Wall time | Detail |
|------|---------------|----------:|--------|
| Sequential | single-thread scan | **113.25 s** | 1,000,000 / 1,000,000 events |
| Parallel (Picard) | 10,000 workers × 100 steps/worker | **3.18 s** | 4 iterations, 1 conflict |

**Speedup: 35.6×** (113.25 s → 3.18 s).

**Correctness:** bit-identical — 1,000,000 fulfillments, **0 mismatches**.

**Observations:**
- With `num_steps × n_workers = 100 × 10,000 = 1,000,000`, the processing window
  spans the entire event horizon, so Picard converged in just **4 iterations**
  (only 1 conflict to resolve across the whole horizon). Fewer, larger iterations
  amortize per-iteration host overhead well.
- This 35.6× is the end-to-end sequential-vs-parallel win at scale, where the
  GPU's per-event parallelism dominates. It is a different measurement from the
  ~1.76× host-orchestration gains in `OPTIMIZATIONS.md`, which isolate host
  overhead on a deliberately host-heavy small-window config.

**Reproduce:**
```bash
# build the 1M-event workload (first 1M events, compact product ids)
python - <<'PY'
import numpy as np, os
src, dst = 'data/npy', 'data/sub1m/npy'; os.makedirs(dst, exist_ok=True)
N = 1_000_000
op   = np.load(src+'/order_products.npy')[:N]
ninf = np.load(src+'/node_index_near_to_far.npy')[:N]
inv  = np.load(src+'/inventory.npy'); cap = np.load(src+'/capacity.npy')
uniq = np.unique(op)
remap = np.full(inv.shape[0], -1, np.int64); remap[uniq] = np.arange(len(uniq))
np.save(dst+'/order_products.npy', remap[op].astype(np.int32))
np.save(dst+'/node_index_near_to_far.npy', ninf.astype(np.int32))
np.save(dst+'/inventory.npy', inv[uniq].astype(inv.dtype))
np.save(dst+'/capacity.npy', cap)
PY

./build/cuda_sim data/sub1m/npy --mode sequential --seed 42 \
    --fulfill_output /tmp/seq1m_fulfill.npy
./build/cuda_sim data/sub1m/npy --mode picard --seed 42 \
    --max_workers 10000 --max_products_per_worker 100 --num_steps 100 \
    --fulfill_output /tmp/par1m_fulfill.npy

python -c "import numpy as np; a=np.load('/tmp/seq1m_fulfill.npy'); \
b=np.load('/tmp/par1m_fulfill.npy'); print('identical:', np.array_equal(a,b))"
```

---

## E2 — Same workload after host optimizations #4–#7

- **Commit:** _(this commit)_ — O(n) `cum_count`, persistent prep buffers,
  `get_slice` as contiguous copies, and OpenMP across the prep loops.
  See `OPTIMIZATIONS.md` §"Scaling to 1,000,000 events".
- **Workload / seed / oracle:** identical to E1 (`data/sub1m/npy`, seed 42).

| Mode | Configuration | Wall time | Detail |
|------|---------------|----------:|--------|
| Sequential | single-thread scan | 113.25 s | unchanged from E1 |
| Parallel (Picard) | 10,000 workers × 100 steps/worker | **1.77 s** | 4 iterations, 1 conflict |

**Speedup: 64.1×** (113.25 s → 1.77 s) — up from 35.6× in E1; the host
optimizations alone took the parallel run **3.18 s → 1.77 s (1.8×)**.

**Correctness:** bit-identical — 1,000,000 fulfillments, **0 mismatches**.

**Phase profile (`PROFILE=1`, per iteration):** kernel 152.8 ms (42.6%), prep
119.7 ms (33.3%), post 63.5 ms (17.7%), H2D 22.7 ms (6.3%), D2H 0.3 ms. `prep`
fell from 491 → 120 ms/iter; the **kernel is now the dominant phase** (its absolute
cost is unchanged — the host work around it shrank). Next targets in
`OPTIMIZATIONS.md` → "Next steps" (P8 warp kernel, P-post, P9 GPU prep).

---

## E3 — Same workload after the warp-cooperative kernel (#8)

- **Commit:** _(this commit)_ — `simulate_worker_kernel_warp`: one warp per worker,
  MLP matvec split across lanes, `logsumexp`/`norm` via warp shuffles. See
  `OPTIMIZATIONS.md` §8.
- **Oracle:** relaxed from bit-identical to **node-decision match rate** (the warp
  reductions reorder float adds). `LEGACY_KERNEL=1` restores the scalar kernel.

| Mode | Configuration | Wall time | Detail |
|------|---------------|----------:|--------|
| Sequential | single-thread scan | 113.25 s | unchanged |
| Parallel (Picard, warp kernel) | 10,000 workers × 100 steps/worker | **1.37 s** | 4 iterations, 1 conflict |

**Speedup: 82.8×** (113.25 s → 1.37 s) — up from 64.1× (E2). The kernel phase fell
**152.8 → 29.1 ms/iter (5.2×)**.

**Correctness (tolerance):** despite non-bit-identical floats, **0 / 1,000,000
node-decision mismatches (100% match)**, iteration count unchanged — the NN enters
only through a `roundf`'d infinitesimal tie-breaker, so last-bit differences are
quantized away. sub100k Config C and A also 0/100,000.

**Phase profile (`PROFILE=1`):** prep 129.1 ms (52.0%), post 67.0 ms (27.0%),
kernel 29.1 ms (11.7%), H2D 22.7 ms (9.2%). Host prep + post are again the
bottleneck (79%); see `OPTIMIZATIONS.md` "Next steps" (P-post, P5b, P9).

---

## E4 — Same workload after the O(window) post phase (#9)

- **Commit:** _(this commit)_ — replace the post-phase `std::sort` of 1M indices
  with an O(window) presence-mark + first-hole scan for `max_t_reset`. See
  `OPTIMIZATIONS.md` §9.
- **Correctness:** `max_t_reset` is reproduced **exactly**. Cross-checked with the
  bit-identical scalar kernel (`LEGACY_KERNEL=1`): Picard+#9 is **0/1,000,000
  mismatches** vs. sequential with iteration count/conflicts unchanged, so #9 does
  not perturb the trajectory. Default warp kernel: 0/1,000,000.

| Mode | Configuration | Wall time | Detail |
|------|---------------|----------:|--------|
| Sequential | single-thread scan | 113.25 s | unchanged |
| Parallel (Picard, warp kernel) | 10,000 workers × 100 steps/worker | **1.09 s** | 4 iterations, 1 conflict |

**Speedup: 104.2×** (113.25 s → 1.09 s) — up from 82.8× (E3); crosses 100×. The
post phase fell **67.0 → 11.1 ms/iter (6×)**.

**Phase profile (`PROFILE=1`):** prep 120.5 ms (65.5%), kernel 29.2 ms (15.9%),
H2D 23.0 ms (12.5%), post 11.1 ms (6.0%). **Host prep is now the lone bottleneck at
65.5%**; next targets in `OPTIMIZATIONS.md` "Next steps" (P5b parallelize prep, P9
GPU-port prep).

### Cumulative speedup vs. sequential (1M events)
| Stage | Wall | Speedup |
|-------|-----:|--------:|
| E1 (`54dcf3e`, #1–#3) | 3.18 s | 35.6× |
| E2 (#4–#7 host prep)  | 1.77 s | 64.1× |
| E3 (#8 warp kernel)   | 1.37 s | 82.8× |
| E4 (#9 post)          | 1.09 s | 104.2× |
| E5 (#10 P5b prep scan) | 0.870 s | 130.2× |
| E6 (#11 init + #12 ntf elim) | 0.756 s | 149.8× |

---

## E5 — Same workload after the parallel/persistent capacity-delta scan (#10)

- **Commit:** _(this commit)_ — `compute_capacity_delta`'s `cumsum` prefix scan made
  persistent (no per-iteration 124 MB alloc+zero) and parallelized across the `np1`
  independent node-columns; `random_assignment`'s assignment scatter parallelized.
  See `OPTIMIZATIONS.md` §10.
- **Correctness:** the column-parallel scan is **bit-identical** to the prior serial
  row-major scan (`x + 0.0f == x`, per-column add order preserved). Cross-checked
  with `LEGACY_KERNEL=1`: Picard+#10 is **0/1,000,000** vs. sequential with iteration
  count/conflicts unchanged. Default warp kernel: 0/1,000,000.

| Mode | Configuration | Wall time | Detail |
|------|---------------|----------:|--------|
| Sequential | single-thread scan | 113.25 s | unchanged |
| Parallel (Picard, warp kernel) | 10,000 workers × 100 steps/worker | **0.870 s** | 4 iterations, 1 conflict |

**Speedup: 130.2×** (113.25 s → 0.870 s) — up from 104.2× (E4). Host prep fell
**120.5 → 51.2 ms/iter (2.35×)**.

**Phase profile (`PROFILE=1`):** prep 51.2 ms (44.8%), kernel 29.2 ms (25.5%), H2D
22.6 ms (19.8%), post 11.1 ms (9.7%), D2H 0.3 ms. Prep is still the top phase but no
longer dominant — kernel, H2D and the remaining prep are now within ~2× of each
other; next targets in `OPTIMIZATIONS.md` "Next steps" (H2D compaction, P9 GPU prep).

---

## E6 — Same workload after init cleanup (#11) + ntf elimination (#12)

- **Commit:** _(this commit)_ — `#11`: a no-init allocator removes the 270 ms serial
  zero-fill of ~600 MB of init scratch (wall-neutral; the cost was page-faulting, not
  the memset — see `OPTIMIZATIONS.md` §11, which also records two reverted negatives:
  `cudaHostRegister` pinning and `MADV_HUGEPAGE`). `#12`: the per-window `ev2d_ntf`
  buffer (124 MB) is eliminated — the kernel gathers the immutable near-to-far
  ordering straight from the device-resident `events.node_index_near_to_far`, indexed
  by the per-slot global event id (`order_2D_index`). See `OPTIMIZATIONS.md` §12.
- **Correctness:** `#12` is **bit-identical** (the gather reads exactly the rows the
  old copy held). `LEGACY_KERNEL=1`: **0/1,000,000** vs. sequential, 4 iters /
  conflicts 1; warp kernel 0/1,000,000; sub100k C and A 0 mismatches.

| Mode | Configuration | Wall time | Detail |
|------|---------------|----------:|--------|
| Sequential | single-thread scan | 113.25 s | unchanged |
| Parallel (Picard, warp kernel) | 10,000 workers × 100 steps/worker | **0.756 s** | 4 iterations, 1 conflict |

**Speedup: 149.8×** (113.25 s → 0.756 s) — up from 130.2× (E5). #12 cut init D2H
events 73 → 2.5 ms, reshape 9.6 → 5.6 ms/iter, and H2D 24.3 → 17.5 ms/iter.

**Note on the floor:** instrumenting `init_iter_context` showed that at 1M / 4
iterations the wall is ~half one-time setup (D2H state + first-touch faulting of the
remaining scratch) and ~half the 4-iteration loop. Faulting can't be sped up here (no
huge pages), so further wins must shrink the footprint (P13: `worker_inv`) or the loop
(P9: GPU prep). See `OPTIMIZATIONS.md` "Next steps".

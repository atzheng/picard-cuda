# Parallel-simulation (Picard) host optimization log

This log tracks optimizations to the **host-side orchestration** of the Picard
iteration (`run_algorithm_day` / `iterate_algorithm` in `src/algorithm.cu`), which
was identified as the dominant bottleneck. Each entry records the diagnosis that
motivated the change, what changed, and the measured incremental improvement.

## Methodology

- **Workload:** `data/sub100k/npy` — first 100,000 events of the full dataset, with
  product ids remapped to a compact range (93,735 distinct products, 30 nodes).
  Generated from `data/npy` (1M products × 30 nodes × 3M events).
- **Metric:** wall-clock `--- total time ---` reported by the binary, **best of 3
  runs** (seed fixed at 42, so iteration count / conflicts are deterministic).
- **Phase profile:** built-in, env-gated. Run with `PROFILE=1` to print a per-phase
  host breakdown (`prep`, `H2D`, `kernel`, `D2H`, `post`) accumulated over all
  Picard iterations. Inert (zero overhead) when `PROFILE` is unset.
- **Two reference configs** (same workload, different parallel shape):
  - **Config C** — `--max_workers 1024 --max_products_per_worker 100 --num_steps 10`
    → 42 iterations. *Host-heavy* (small window per iteration, many iterations);
    best exposes per-iteration orchestration cost.
  - **Config A** — `--max_workers 256 --max_products_per_worker 400 --num_steps 100`
    → 11 iterations. More kernel-bound.
- **Correctness oracle:** at convergence the Picard result must be **bit-identical**
  to the single-threaded sequential scan (`--mode sequential`) on the same data —
  two independent code paths that only agree if the math is preserved.
  Checked with `np.array_equal` after every change.

## Results summary

All numbers best-of-3, measured together in a single run for consistency.

| # | Optimization | Config C (42 it) | Config A (11 it) | vs prev (C) | vs baseline (C) |
|---|--------------|-----------------:|-----------------:|------------:|----------------:|
| 0 | Baseline (original)        | 0.521 s | 0.285 s | —     | 1.00× |
| 1 | State-resident host loop   | 0.372 s | 0.252 s | 1.40× | 1.40× |
| 2 | Window-scoped inventory reshape | 0.303 s | 0.236 s | 1.23× | 1.72× |
| 3 | Window-scoped conflict/commit   | 0.296 s | 0.232 s | 1.02× | 1.76× |
| 4–7 | O(n) cum_count + persistent buffers + get_slice copies + OpenMP | 0.247 s | 0.200 s | 1.20× | **2.11×** |

(#4–#7 were driven by the **1M-event** workload where `prep` dominates — see the
"Scaling to 1,000,000 events" section below for the per-step phase breakdown; the
sub100k columns here are the cross-check that they stay bit-identical and still help.)

---

## 0. Baseline (original)

The starting point. `iterate_algorithm` was fully self-contained: every Picard
iteration pulled the **entire** simulation state and **all** event arrays from the
GPU, did the orchestration on the host, pushed everything back, and `malloc`/`free`d
all device scratch each call.

Per-iteration PCIe traffic (Config C dims, 93,735 products × 31, 100k events):
- **D2H in:** capacity + inventory (~11.6 MB) + fulfill + event_products +
  event_quantities + event_ntf (~12.4 MB) ≈ **25 MB**
- **H2D out (writeback):** capacity + inventory (~11.6 MB) + fulfill ≈ **12 MB**
- Plus a fresh `cudaMalloc`/`cudaFree` of every scratch buffer each iteration.

So each iteration moved **~37 MB of full-state traffic** (on top of the unavoidable
`worker_inv` upload) to advance a window of only `num_steps × n_workers` events.

**Measured:** Config C **0.528 s**, Config A **0.285 s**.

---

## 1. State-resident host loop

**Diagnosis.** The per-iteration cost scaled with *total problem size*, not the
*window* actually processed. The immutable event arrays (`product`, `quantity`,
`node_index_near_to_far`) never change yet were re-downloaded every iteration; the
mutable state was round-tripped in full even though the host is the authoritative
owner of it during the loop.

**Change.** Introduced `IterContext`:
- Immutable event arrays copied to host **once** before the loop.
- Mutable state (`capacity`, `inventory`, `fulfill`) kept **resident on the host**
  across all iterations; only the small per-step capacity vector is synced to the
  device for the kernel. State is flushed back to the device **once** after the loop.
- Device scratch buffers (`d_worker_inv`, `d_ev_*`, …) allocated **once** and reused.

`run_algorithm_day` now does: `init_iter_context` (stage in once) → loop
`iterate_core` (no full-state PCIe) → `finalize_iter_context` (flush once).

**Result.** PCIe is removed from the hot path: in the profile, D2H drops to ~0.2%
and H2D to 4–10% (only the unavoidable `worker_inv` upload remains).

| Config | baseline | tier1 | speedup |
|--------|---------:|------:|--------:|
| C (42 it) | 0.528 s | **0.367 s** | **1.44×** |
| A (11 it) | 0.285 s | **0.244 s** | **1.17×** |

Correctness: Picard == sequential, 0 mismatches.

**Profile after (Config C, `PROFILE=1`):**
```
prep (CPU)     49.3%   (of which reshape 23.0%)
H2D upload      9.7%
kernel         31.8%
D2H readback    0.2%
post (CPU)      9.0%
```
→ Next bottleneck is **prep (CPU)**, dominated by rebuilding the full worker
assignment and scattering the full inventory every iteration (next entry).

---

## 2. Window-scoped inventory reshape

**Diagnosis.** With PCIe gone, `prep` (the host construction of the 2D worker
problem) became the top cost, and within it the **inventory reshape** was the
single biggest piece (23% of total, 1.98 ms/iter in Config C). The old reshape
zeroed the entire persistent `worker_inv` buffer (`n_workers × max_ppw × np1` ≈
3.2 M floats) and scattered **all `n_products` rows** (93,735 × 31 ≈ 2.9 M writes)
into it every iteration — even though the kernel only ever reads the rows of the
≤ `num_steps × n_workers` products that appear in the current window.

**Change.** Fold the reshape into the existing window-sized event-build loop:
scatter **only the inventory row of each valid event** (one row per valid 2D slot),
straight from the authoritative host inventory. Also **drop the full-buffer
zeroing**: every slot the kernel reads is overwritten here before upload, and
untouched slots are never read, so leaving them stale is safe. This turns an
`O(n_products · np1)` scatter + `O(n_workers · max_ppw · np1)` memset into an
`O(window · np1)` scatter.

**Result.** Reshape cost 1.98 → 0.60 ms/iter; `prep` 4.25 → 2.46 ms/iter.

| Config | prev | opt2 | speedup |
|--------|-----:|-----:|--------:|
| C (42 it) | 0.372 s | **0.303 s** | **1.23×** |
| A (11 it) | 0.252 s | **0.236 s** | 1.07× |

Correctness: Picard == sequential, 0 mismatches (both configs); iteration counts
unchanged (42 / 11), confirming the optimization preserves the exact trajectory.

---

## 3. Window-scoped conflict detection & commit

**Diagnosis.** The post-kernel bookkeeping still did three `O(n_events)` passes per
iteration: a full `new_fulfill = h_fulfill` copy, an `n_events`-sized `conflicts`
vector + fill, and a linear scan for the first conflict — plus a final full
`h_fulfill = new_fulfill` assign. The new fulfillment can only differ from the
committed one at the assigned window indices, so all of this is window-sized work
dressed up as full-array work.

**Change.** The first conflict is the lowest assigned window index whose new value
differs from the old, so scan the **window** (reading the old `h_fulfill` value
before overwriting), then commit the window's fulfillments **in place** into the
resident `h_fulfill` (no full copy). The inventory/capacity update loops directly
over `[t_reset, new_t_reset)` (bounded by `n_events`). Removes the per-iteration
`O(n_events)` allocation and three scans.

**Result.** `post` 0.695 → 0.445 ms/iter.

| Config | prev | opt3 | speedup |
|--------|-----:|-----:|--------:|
| C (42 it) | 0.303 s | **0.296 s** | 1.02× |
| A (11 it) | 0.236 s | **0.232 s** | 1.02× |

Modest, as expected (`post` was ~10%), but it removes the last place where
per-iteration host cost scaled with total `n_events` rather than the window.

Correctness: Picard == sequential, 0 mismatches (both configs); iteration counts
unchanged.

**Profile after (Config C, `PROFILE=1`):**
```
prep (CPU)     37.6%   (reshape/event-build 9.1%)
H2D upload     13.8%   (worker_inv upload — 12.7 MB/iter)
kernel         41.6%
D2H readback    0.2%
post (CPU)      6.8%
```
(The `of which:reshape` line now measures the combined event-build + scatter loop.)

---

## Scaling to 1,000,000 events (#4–#7)

Optimizations #1–#3 were tuned on the host-heavy sub100k Config C. At the
realistic **1M-event** workload (`data/sub1m/npy`, 567,323 products, 10,000
workers × 100 steps, 4 iterations — see `experiments.md`) `prep` (single-threaded
CPU) was **67.6%** of runtime while 23 host cores and the A100 sat idle. #4–#7
attack that. Starting point is commit `54dcf3e` (i.e. #1–#3 already in): **3.16 s**.

| # | Optimization | prep ms/iter | 1M wall | sub100k C | sub100k A |
|---|--------------|-------------:|--------:|----------:|----------:|
| 3 | (starting point, #1–#3) | 491 | 3.16 s | 0.296 s | 0.232 s |
| 4 | O(n) `cum_count`        | 407 | 2.83 s |    —     |    —     |
| 5–7 | persistent buffers + `get_slice` copies + OpenMP | **120** | **1.77 s** | **0.247 s** | **0.200 s** |

**Cumulative vs. sequential (1M):** 113.25 s → **1.77 s = 64.1×** (was 35.6× at
`54dcf3e`). All steps **bit-identical** to the sequential scan (0 mismatches);
iteration counts unchanged.

### 4. O(n) `cum_count` (drop the 1M-element `stable_sort`)

**Diagnosis.** `cum_count` (`utils.h`) computes, per event, how many earlier
events share its worker — via a full `std::stable_sort` of all `eff_num_steps`
(1M) elements plus rank/bincount passes. A microbench put that sort at **93.3 ms**
of the 491 ms prep.

**Change.** The same value is a one-pass running counter:
`result[i] = count[workers[i]]++` over an `O(n_workers)` array. Integer →
bit-identical. Microbench: **93.3 ms → 2.3 ms (41×)** on the sub-step.

**Result.** prep 491 → 407 ms/iter; 1M wall 3.16 → 2.83 s. 0 mismatches.

### 5–7. Persistent buffers + `get_slice` copies + OpenMP

Three changes, landed and measured together (they interlock — OpenMP needs the
stable buffers, and the per-element loops are the parallel targets):

- **#6 Persistent scratch.** ~15 `O(window)`/`O(n_products)` vectors were
  heap-allocated every `iterate_core` call (`slice_*`, `product_*`, `order_2D_*`,
  `valid_mask`, `capacity_delta`, `ev2d_*`, `fulfill_2D`, `order_flat`, …). Hoisted
  into `IterContext`, sized once, reused. Removes per-iteration alloc + zero-fill +
  page-fault churn and gives OpenMP fixed buffers to write into.
- **#7 `get_slice` as two copies.** The window roll was a per-element double
  modulo over three 1M arrays (6M `%`/iter). It is a rotation, so it is now two
  contiguous `std::copy` ranges (`k = (t_reset − t0) mod eff`).
- **#5 OpenMP over prep.** `#pragma omp parallel for` on the embarrassingly
  parallel prep loops (24 cores): event→worker map, `order_2D_index` scatter
  (unique `(w,s)` ⇒ race-free), `valid_mask`/`order_2D_relative`, the inventory
  reshape/event-build, and the **outer `w` loop** of `compute_capacity_delta`.
  The cumsum prefix scan inside it stays **serial** (a parallel scan would reorder
  float adds and break bit-identicality); only the independent gather is
  parallelized. `cum_count` (now O(n), ~2 ms) and the Fisher–Yates shuffle stay
  serial. Built with `-Xcompiler -fopenmp` (CMake `OpenMP::OpenMP_CXX`).

**Result.** prep 407 → **120 ms/iter** (3.4×); reshape 69 → 10 ms/iter; 1M wall
2.83 → **1.77 s**. sub100k C 0.296 → 0.247 s, A 0.232 → 0.200 s. 0 mismatches,
iteration counts unchanged.

**Profile after (1M, `PROFILE=1`):**
```
kernel         42.6%   152.8 ms/iter   <-- now the top phase
prep (CPU)     33.3%   119.7 ms/iter
post (CPU)     17.7%    63.5 ms/iter   (dominated by a 1M-element std::sort)
H2D upload      6.3%    22.7 ms/iter
D2H readback    0.1%     0.3 ms/iter
```

---

## Next steps (diagnosed, not yet implemented)

P4–P7 are **done** (see #4–#7 above). After them the 1M profile is:

```
kernel         42.6%   152.8 ms/iter   <-- now the top phase (unchanged in abs.)
prep (CPU)     33.3%   119.7 ms/iter
post (CPU)     17.7%    63.5 ms/iter
H2D upload      6.3%    22.7 ms/iter
D2H readback    0.1%     0.3 ms/iter
```

### P8 — Warp-parallel kernel (now the top phase, 42.6%)
Launched `<<<n_workers, 1>>>` — one thread per block, so 31/32 of each warp is idle
and the per-event NN inference runs fully serial. A warp-per-worker design
parallelizes the per-event matvec/feasibility scan. **Caveat:** changes the NN's
floating-point reduction order → breaks the *bit-identical* oracle; needs a
tolerance-based oracle or a fixed reduction order first. Highest payoff now,
highest care. **Needs a product decision on the oracle before starting.**

### P-post — Drop the 1M-element `std::sort` in `post` (17.7%)
`post` is now dominated by sorting `order_flat` (all `n_workers·num_steps` global
indices) every iteration just to find `max_t_reset` — the same "needless sort"
pattern P4 removed from prep. Event 0 always maps to `t_reset`, so the contiguous
run starts there; an `O(window)` "present" bitmap over `[t_reset, t_reset+eff)` +
a first-gap scan replaces the `O(window log window)` sort. Integer; must reproduce
`max_t_reset` exactly — verify against the sort on the trajectory before trusting.

### P5b — OpenMP / parallel-scan the `compute_capacity_delta` cumsum
The serial prefix scan is the remaining un-parallelized chunk of prep. A
blocked parallel scan would speed it up but reorders float adds (bit-identicality
risk); gated on the same oracle decision as P8.

### P9 — Move prep onto the GPU (architectural, highest ceiling)
prep is still 33% of runtime and is pure array transforms (gather/scatter,
counting, cumsum) on data already bound for the device. Porting it to CUDA removes
the CPU from the hot path entirely. Largest effort; cumsum carries the same
float-order risk as P5b.

### Projected trajectory (1M-event wall clock)
| Step | wall | note |
|------|-----:|------|
| `54dcf3e` (#1–#3) | 3.16 s | starting point |
| **+#4–#7 (done)** | **1.77 s** | **bit-identical, 64.1× vs sequential** |
| +P-post | ~1.5 s | bit-identical (verify max_t_reset) |
| +P8 (kernel) | ~1.0 s | needs oracle change |
| +P9 (GPU prep) | well under 1 s | architectural |

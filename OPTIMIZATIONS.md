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
| 3 | Window-scoped conflict/commit   | 0.296 s | 0.232 s | 1.02× | **1.76×** |

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

## Next steps (diagnosed, not yet implemented)

### The bottleneck inverts at scale

The Config-C "next steps" below were diagnosed on a deliberately host-heavy
*small-window* config. At the **realistic 1M-event scale** (`data/sub1m/npy`,
10,000 workers × 100 steps, 4 iterations — see `experiments.md` E1), the phase mix
is completely different:

```
prep (CPU)     67.6%   491.3 ms/iter   <-- now totally dominant
  of which reshape  10.1%  73.8 ms/iter
kernel         21.0%   152.6 ms/iter
post (CPU)      8.3%    60.7 ms/iter
H2D upload      2.8%    20.5 ms/iter
D2H readback    0.3%     2.1 ms/iter
```

**`prep` is 67.6% and it is single-threaded CPU work running while a 24-core host
and an A100 sit idle.** That, not the kernel, is where the next "faster relative to
sequential" wins are. Plan, in priority order:

### P4 — O(n) `cum_count` (replace the 1M-element `stable_sort`) — *do first*
`cum_count` (utils.h) computes, for each event, how many earlier events share its
worker — currently via a **full `std::stable_sort` of all `eff_num_steps` (1M)
elements** plus rank/bincount passes. That sort is the single biggest sub-cost.
The same result is a one-pass running counter: `result[i] = count[workers[i]]++`
over an `O(n_workers)` counter array. **Integer math → bit-identical.**
- **Measured (microbench, 1M events):** stable_sort path **93.3 ms** → counter
  path **2.3 ms** = **41× on this sub-step**, ~91 ms/iter saved (~19% of total
  runtime, ~0.36 s off the 3.16 s wall at 4 iterations).
- Risk: none. Drop-in, integer, exact. **Highest ROI / lowest risk — start here.**

### P5 — OpenMP across prep's element-wise loops (24 idle cores)
After P4, the remaining ~400 ms/iter of prep is mostly embarrassingly parallel
per-element work: the three `get_slice` loops, the worker mapping, `valid_mask`,
`order_2D_relative`, `order_2D_index` (writes are unique per `(w,s)` so no race),
the event-build + inventory-scatter loop, and the **outer `w` loop of
`compute_capacity_delta`**. All integer or independent-row float → bit-identical
under `#pragma omp parallel for`.
- **Caveat:** the *cumsum* (prefix scan) inside `compute_capacity_delta` is a
  sequential dependency; keep it serial (or use a parallel scan, which reorders
  float adds and would break bit-identicality). Parallelize only the gather/diff
  loop that follows it.
- **Estimated:** if ~75% of post-P4 prep parallelizes at ~8×, prep drops from
  ~400 to ~140 ms/iter — the largest single remaining win.

### P6 — Persistent prep scratch buffers
Every `iterate_core` call heap-allocates ~15 vectors sized `O(window)` /
`O(n_products)` (`slice_*`, `product_worker_map`, `workers`, `order_2D_*`,
`valid_mask`, `capacity_delta`, `ev2d_*`, …). Hoist them into `IterContext`,
allocate once, reuse. Removes per-iteration allocator + zero-fill + page-fault
cost, and is a **prerequisite for P5** (stable buffers to parallelize over).

### P7 — Strength-reduce `get_slice` (kill the double modulo)
`get_slice_int` does two `%` per element across three 1M arrays (6M modulos/iter).
The roll is a rotation: emit it as two contiguous `std::copy` ranges instead.
Integer, exact, easy.

### P8 — Warp-parallel kernel (now only 21%, but the next ceiling after prep)
Kernel is launched `<<<n_workers, 1>>>` (one thread/block; 31/32 of each warp
idle). A warp-per-worker design parallelizes the per-event NN matvec/feasibility
scan. **Caveat (unchanged):** changes float reduction order → breaks the
bit-identical oracle; needs a tolerance oracle or a fixed reduction order first.
Deprioritized vs. prep because it is only 21% at scale.

### P9 — Move prep onto the GPU (architectural, highest ceiling)
Since prep is 68% and pure array transforms (gather/scatter, counting, cumsum) on
data that is *already going to the device*, porting it to CUDA removes the CPU from
the hot path entirely. Largest effort and the cumsum carries the same float-order
risk as P5; sequence it after P4–P7 have proven the host-side wins.

### Projected trajectory (1M-event wall clock, 3.16 s today)
| Step | prep ms/iter | est. wall | note |
|------|-------------:|----------:|------|
| now (54dcf3e) | 491 | 3.16 s | — |
| +P4 | ~400 | ~2.8 s | bit-identical, trivial |
| +P5/P6/P7 | ~140 | ~1.8 s | bit-identical, OpenMP |
| +P8 (kernel) | ~140 | ~1.4 s | needs oracle change |
| +P9 (GPU prep) | — | well under 1 s | architectural |

(Estimates are back-of-envelope from the phase profile; each will be measured and
logged as an entry above when implemented.)

"""
Data ingestion (S3) + multi-scale Picard benchmarks.

Pipeline:
    download_raw     -> sync the raw pickle problem from S3
    convert_npy      -> tests/convert_pickles_to_npy.py: pickles -> .npy (full product range)
    compact_npy      -> remap product ids to a compact range (same recipe as experiments.md)
    build            -> cmake build of ./build/cuda_sim
    sequential_baseline -> single-thread scan (correctness oracle + speedup denominator)
    picard_benchmark -> one Picard run per worker count in WORKER_COUNTS
    summary          -> results/benchmark_summary.csv (timings, speedup, mismatches)
    benchmark_perstep -> `--mode benchmark` per-event policy vs. state-update timing
    plot             -> results/benchmark_plot.png (runtime & speedup vs. num workers)

Usage:
    snakemake -j 8 --config s3_raw_uri=s3://bucket/path
    snakemake -j 8                      # uses the defaults below

Requirements on the run machine:
    - `aws` CLI configured for the source bucket (this repo's dev env talks to a
      Cloudflare R2 bucket via AWS_ENDPOINT_URL + AWS_ACCESS_KEY_ID/SECRET; aws-cli
      >= 2.13 picks up AWS_ENDPOINT_URL automatically).
    - python3 with numpy + pgeocode (see requirements-benchmark.txt) for the pickle
      -> npy conversion (pgeocode geocodes zips; needs network access on first run
      to fetch its GeoNames cache).
    - cmake + CUDA toolkit + OpenMP to build cuda_sim.
"""
import math

# ---------------------------------------------------------------------------
# Config — override any of these with `--config key=value` on the CLI.
# ---------------------------------------------------------------------------
config.setdefault(
    "s3_raw_uri",
    "s3://research/picard/data/"
    "days=1:orders=3000000:products=1000000:nodes=30:fulfillable=0.8:alpha=1.0:seed=0",
)
config.setdefault("num_products", 1000000)   # canonical product count for this dataset
config.setdefault("n_events", 3000000)       # events in the raw dataset
config.setdefault("worker_counts", [100, 1000, 10000, 100000])
config.setdefault("window_multiplier", 1.7)  # workers*steps ~= window_multiplier*n_events
                                              # (see experiments.md E10: this minimizes
                                              #  Picard iteration count on this workload)
config.setdefault("seed", 42)
config.setdefault("perstep_num_steps", 1000000)  # events replayed by `--mode benchmark`

WORKERS = [int(w) for w in config["worker_counts"]]
N_EVENTS = int(config["n_events"])
N_PRODUCTS = int(config["num_products"])
WINDOW_MULT = float(config["window_multiplier"])
SEED = int(config["seed"])
PERSTEP_NUM_STEPS = int(config["perstep_num_steps"])


def steps_for(workers):
    """num_steps so that workers*steps ~= WINDOW_MULT * n_events (min-iteration window)."""
    return max(1, math.ceil(WINDOW_MULT * N_EVENTS / workers))


def mppw_for(workers):
    """max_products_per_worker with a 10% margin over workers*mppw >= n_products."""
    return max(1, math.ceil(1.1 * N_PRODUCTS / workers))


rule all:
    input:
        "results/benchmark_summary.csv",
        "results/perstep_benchmark.txt",
        "results/benchmark_plot.png",


# ---------------------------------------------------------------------------
# 1. Data ingestion from S3
# ---------------------------------------------------------------------------
rule download_raw:
    output:
        orders="data/raw/orders.p",
        inventory="data/raw/inventory_dict_original.p",
        capacity="data/raw/daily_capacity_dict.p",
        nodes="data/raw/nodes.p",
    params:
        uri=config["s3_raw_uri"],
    shell:
        """
        mkdir -p data/raw
        aws s3 cp "{params.uri}/orders.p" {output.orders}
        aws s3 cp "{params.uri}/inventory_dict_original.p" {output.inventory}
        aws s3 cp "{params.uri}/daily_capacity_dict.p" {output.capacity}
        aws s3 cp "{params.uri}/nodes.p" {output.nodes}
        """


# ---------------------------------------------------------------------------
# 2. Convert raw pickles -> .npy (full, uncompacted product range)
# ---------------------------------------------------------------------------
rule convert_npy:
    input:
        orders="data/raw/orders.p",
        inventory="data/raw/inventory_dict_original.p",
        capacity="data/raw/daily_capacity_dict.p",
        nodes="data/raw/nodes.p",
    output:
        inventory="data/npy_full/inventory.npy",
        capacity="data/npy_full/capacity.npy",
        order_products="data/npy_full/order_products.npy",
        order_dates="data/npy_full/order_dates.npy",
        ntf="data/npy_full/node_index_near_to_far.npy",
    params:
        num_products=N_PRODUCTS,
    shell:
        """
        python3 tests/convert_pickles_to_npy.py data/raw data/npy_full \
            --num_products {params.num_products} --suffix ""
        """


# ---------------------------------------------------------------------------
# 3. Remap product ids to a compact range (same recipe used for data/sub*m in
#    experiments.md, so results are directly comparable to E1/E10).
# ---------------------------------------------------------------------------
rule compact_npy:
    input:
        inventory="data/npy_full/inventory.npy",
        capacity="data/npy_full/capacity.npy",
        order_products="data/npy_full/order_products.npy",
        ntf="data/npy_full/node_index_near_to_far.npy",
    output:
        inventory="data/full/npy/inventory.npy",
        capacity="data/full/npy/capacity.npy",
        order_products="data/full/npy/order_products.npy",
        ntf="data/full/npy/node_index_near_to_far.npy",
    run:
        import numpy as np, os
        os.makedirs("data/full/npy", exist_ok=True)
        op = np.load(input.order_products)
        ninf = np.load(input.ntf)
        inv = np.load(input.inventory)
        cap = np.load(input.capacity)
        uniq = np.unique(op)
        remap = np.full(inv.shape[0], -1, np.int64)
        remap[uniq] = np.arange(len(uniq))
        np.save(output.order_products, remap[op].astype(np.int32))
        np.save(output.ntf, ninf.astype(np.int32))
        np.save(output.inventory, inv[uniq].astype(inv.dtype))
        np.save(output.capacity, cap)


# ---------------------------------------------------------------------------
# 4. Build the CUDA binary
# ---------------------------------------------------------------------------
rule build:
    output:
        "build/cuda_sim",
    shell:
        """
        mkdir -p build
        cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j
        """


# ---------------------------------------------------------------------------
# 5. Sequential baseline: correctness oracle + speedup denominator
# ---------------------------------------------------------------------------
rule sequential_baseline:
    input:
        bin="build/cuda_sim",
        order_products="data/full/npy/order_products.npy",
    output:
        csv="results/sequential.csv",
        fulfill="results/sequential_fulfill.npy",
    params:
        seed=SEED,
    shell:
        """
        mkdir -p results
        ./{input.bin} data/full/npy --mode sequential --seed {params.seed} \
            --output {output.csv} --fulfill_output {output.fulfill}
        """


# ---------------------------------------------------------------------------
# 6. Picard benchmark, one run per worker count
# ---------------------------------------------------------------------------
rule picard_benchmark:
    input:
        bin="build/cuda_sim",
        order_products="data/full/npy/order_products.npy",
    output:
        csv="results/picard_w{workers}.csv",
        fulfill="results/picard_w{workers}_fulfill.npy",
    params:
        seed=SEED,
        steps=lambda wc: steps_for(int(wc.workers)),
        mppw=lambda wc: mppw_for(int(wc.workers)),
    shell:
        """
        mkdir -p results
        ./{input.bin} data/full/npy --mode picard --seed {params.seed} \
            --max_workers {wildcards.workers} \
            --max_products_per_worker {params.mppw} \
            --num_steps {params.steps} \
            --output {output.csv} --fulfill_output {output.fulfill}
        """


# ---------------------------------------------------------------------------
# 7. Summarize + correctness check (parallel fulfillment vs. sequential)
# ---------------------------------------------------------------------------
rule summary:
    input:
        seq_csv="results/sequential.csv",
        seq_fulfill="results/sequential_fulfill.npy",
        picard_csv=expand("results/picard_w{workers}.csv", workers=WORKERS),
        picard_fulfill=expand("results/picard_w{workers}_fulfill.npy", workers=WORKERS),
    output:
        "results/benchmark_summary.csv",
    run:
        import csv
        import numpy as np

        seq_row = next(csv.DictReader(open(input.seq_csv)))
        seq_time = float(seq_row["run_time_1"])
        seq_fulfill = np.load(input.seq_fulfill)

        rows = []
        for w in WORKERS:
            row = next(csv.DictReader(open(f"results/picard_w{w}.csv")))
            par_fulfill = np.load(f"results/picard_w{w}_fulfill.npy")
            mismatches = int(np.sum(par_fulfill != seq_fulfill))
            wall = float(row["run_time_1"])
            rows.append({
                "workers": w,
                "num_steps": steps_for(w),
                "max_products_per_worker": mppw_for(w),
                "window": w * steps_for(w),
                "iterations": row["iterations_1"],
                "conflicts": row["conflicts"],
                "wall_time_s": wall,
                "sequential_time_s": seq_time,
                "speedup": seq_time / wall,
                "mismatches": mismatches,
                "n_events": len(seq_fulfill),
            })

        with open(output[0], "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


# ---------------------------------------------------------------------------
# 8. Per-step micro-benchmark: policy cost vs. state-update cost, single
#    thread (`--mode benchmark`; independent of the worker sweep above).
# ---------------------------------------------------------------------------
rule benchmark_perstep:
    input:
        bin="build/cuda_sim",
        order_products="data/full/npy/order_products.npy",
    output:
        "results/perstep_benchmark.txt",
    params:
        num_steps=PERSTEP_NUM_STEPS,
    shell:
        """
        mkdir -p results
        ./{input.bin} data/full/npy --mode benchmark --num_steps {params.num_steps} \
            | tee {output}
        """


# ---------------------------------------------------------------------------
# 9. Plot: wall time & speedup vs. num workers
# ---------------------------------------------------------------------------
rule plot:
    input:
        summary="results/benchmark_summary.csv",
    output:
        "results/benchmark_plot.png",
    shell:
        """
        python3 scripts/plot_benchmark.py {input.summary} {output}
        """

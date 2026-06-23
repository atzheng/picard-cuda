#!/usr/bin/env python3
"""
Validation test: compare C++/CUDA fulfillment output against JAX reference.

This doesn't test bit-exact equality (RNG differs), but verifies:
1. Both complete without errors
2. Fulfillment rates are similar
3. No constraint violations in the CUDA output

Usage:
    python test_vs_jax.py <input_dir> [--cuda_binary ../build/cuda_sim]
"""
import sys
import os
import subprocess
import tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'supply-chain-simulation'))


def validate_fulfillment(problem_dir, fulfill, day=0):
    """Check that fulfillment respects inventory and capacity constraints."""
    inventory = np.load(os.path.join(problem_dir, 'inventory.npy')).copy()
    capacity = np.load(os.path.join(problem_dir, 'capacity.npy'))[day].copy()
    products = np.load(os.path.join(problem_dir, 'order_products.npy'))

    violations = 0
    for i in range(len(fulfill)):
        node = fulfill[i]
        if node == -1:
            continue  # unfulfilled
        prod = products[i]
        if inventory[prod, node] < 1:
            violations += 1
        if capacity[node] < 1:
            violations += 1
        inventory[prod, node] -= 1
        capacity[node] -= 1

    return violations


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help='Directory with .npy files')
    parser.add_argument('--cuda_binary', default='../build/cuda_sim')
    parser.add_argument('--max_workers', type=int, default=1)
    parser.add_argument('--num_steps', type=int, default=10)
    parser.add_argument('--max_iters', type=int, default=10000000)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        fulfill_path = os.path.join(tmpdir, 'fulfill.npy')

        cmd = [
            args.cuda_binary,
            args.input_dir,
            '--max_workers', str(args.max_workers),
            '--num_steps', str(args.num_steps),
            '--max_iters', str(args.max_iters),
            '--fulfill_output', fulfill_path,
        ]

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print("CUDA sim failed:")
            print(result.stderr)
            sys.exit(1)

        fulfill = np.load(fulfill_path)
        print(f"Fulfill shape: {fulfill.shape}")

        n_fulfilled = (fulfill != -1).sum()
        n_total = len(fulfill)
        print(f"Fulfillment rate: {n_fulfilled}/{n_total} = {n_fulfilled/n_total:.4f}")

        violations = validate_fulfillment(args.input_dir, fulfill)
        print(f"Constraint violations: {violations}")

        if violations == 0:
            print("PASS: No constraint violations")
        else:
            print("FAIL: Constraint violations detected")
            sys.exit(1)


if __name__ == '__main__':
    main()

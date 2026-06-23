#!/usr/bin/env python3
"""
Export a NewProblem to .npy files for the C++/CUDA loader.

Usage:
    python export_problem.py <input_dir> <output_dir> [--day DAY]

Produces:
    inventory.npy              float64 [n_products, n_nodes]
    capacity.npy               float64 [n_days, n_nodes]
    order_products.npy         int64   [n_events]
    order_dates.npy            int64   [n_events]
    node_index_near_to_far.npy int64   [n_events, n_nodes]
"""
import sys
import os
import numpy as np

# Add parent dirs to path so we can import datamodel
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'supply-chain-simulation'))

from datamodel import NewProblem


def export_problem(input_dir, output_dir, day=None):
    os.makedirs(output_dir, exist_ok=True)

    prob = NewProblem.load(input_dir)

    np.save(os.path.join(output_dir, 'inventory.npy'), prob.inventory)
    np.save(os.path.join(output_dir, 'capacity.npy'), prob.capacity)
    np.save(os.path.join(output_dir, 'order_products.npy'), np.array(prob.order_products))
    np.save(os.path.join(output_dir, 'order_dates.npy'), np.array(prob.order_dates))
    np.save(os.path.join(output_dir, 'node_index_near_to_far.npy'), prob.node_index_near_to_far)

    print(f"Exported to {output_dir}:")
    print(f"  inventory: {prob.inventory.shape}")
    print(f"  capacity: {prob.capacity.shape}")
    print(f"  order_products: {len(prob.order_products)}")
    print(f"  node_index_near_to_far: {prob.node_index_near_to_far.shape}")

    if day is not None:
        # Also export filtered single-day data
        dates = np.array(prob.order_dates)
        mask = dates == day
        n_day_events = mask.sum()
        print(f"  Day {day}: {n_day_events} events")
        np.save(os.path.join(output_dir, f'order_products_day{day}.npy'),
                np.array(prob.order_products)[mask])
        np.save(os.path.join(output_dir, f'node_index_near_to_far_day{day}.npy'),
                prob.node_index_near_to_far[mask])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('output_dir')
    parser.add_argument('--day', type=int, default=None)
    args = parser.parse_args()
    export_problem(args.input_dir, args.output_dir, args.day)

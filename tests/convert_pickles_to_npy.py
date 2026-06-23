#!/usr/bin/env python3
"""
Self-contained reconstruction of datamodel.NewProblem.load + export_problem.py.

The original pipeline needs the `datamodel` / `fulfillment_optimization` /
`synthetic_data` modules plus a population CSV for zip->lat/long. Those aren't
available here, so this script:

  * loads the raw pickle problem (inventory/capacity/orders/nodes) directly, and
  * geocodes zips with pgeocode (offline GeoNames) instead of the population CSV,
  * computes distances with the SAME mpu haversine formula (radius=6371 km) plus
    the SAME np.random.exponential(0.1) noise seeded at 0 (row-major: order, node),

producing the exact .npy files the C++/CUDA loader expects.

Usage:
    convert_pickles_to_npy.py <raw_dir> <out_dir> --num_products N [--suffix _0]
"""
import os
import sys
import argparse
import pickle

import numpy as np
import pgeocode


def load_pickles(raw_dir, suffix):
    def L(name):
        return pickle.load(open(os.path.join(raw_dir, f"{name}{suffix}.p"), "rb"))
    inventory = L("inventory_dict_original")   # {product_name: {node_id: qty}}
    capacity = L("daily_capacity_dict")        # {date: {node_id: cap}}
    orders_by_date = L("orders")               # {date: [order dicts]}
    nodes = L("nodes")                         # [ {node_id, zip_code, ...} ]
    # Flatten orders across dates, preserving date order then list order.
    orders = []
    for d in sorted(orders_by_date.keys()):
        orders.extend(orders_by_date[d])
    return inventory, capacity, orders, nodes


def extract_product_id(name):
    return int(str(name).split("_")[1])


def geocode_zips(zips):
    """Return dict zip(str, 5-digit zero-padded) -> (lat, lon) via pgeocode."""
    nomi = pgeocode.Nominatim("us")
    uniq = sorted(set(str(z).zfill(5) for z in zips))
    df = nomi.query_postal_code(uniq)
    out = {}
    for z, lat, lon in zip(uniq, df["latitude"].to_numpy(), df["longitude"].to_numpy()):
        out[z] = (lat, lon)
    return out


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized mpu.haversine_distance (radius=6371 km)."""
    radius = 6371.0
    lat1r = np.radians(lat1); lat2r = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return radius * c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--num_products", type=int, required=True,
                    help="canonical_num_products (from the dataset name)")
    ap.add_argument("--suffix", default="_0",
                    help="pickle file suffix for sample path (default _0)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading pickles...", flush=True)
    inventory, capacity, orders, nodes = load_pickles(args.raw_dir, args.suffix)
    num_products = args.num_products
    num_nodes = len(nodes)
    n_events = len(orders)
    node_id_to_int = {node["node_id"]: i for i, node in enumerate(nodes)}
    print(f"  {num_products} products (canonical), {num_nodes} nodes, "
          f"{n_events} orders", flush=True)

    # --- order_dates / order_products ---
    init_date = orders[0]["timestamp"].date().toordinal()
    order_dates = np.array(
        [o["timestamp"].date().toordinal() - init_date for o in orders],
        dtype=np.int64)
    order_products = np.array(
        [extract_product_id(o["product"]) for o in orders], dtype=np.int64)

    # --- inventory [num_products, num_nodes] ---
    print("Building inventory matrix...", flush=True)
    inv_out = np.zeros((num_products, num_nodes), dtype=np.float64)
    for product, node_inv in inventory.items():
        pid = extract_product_id(product)
        for node_id, inv in node_inv.items():
            inv_out[pid, node_id_to_int[node_id]] = inv

    # --- capacity [n_days, num_nodes] ---
    n_days = int(order_dates[-1]) + 1
    cap_out = np.zeros((n_days, num_nodes), dtype=np.float64)
    for d, node_cap in capacity.items():
        di = d.toordinal() - init_date
        if 0 <= di < n_days:
            for node_id, cap in node_cap.items():
                cap_out[di, node_id_to_int[node_id]] = cap

    # --- distances -> node_index_near_to_far [n_events, num_nodes] ---
    print("Geocoding zips...", flush=True)
    node_zips = [str(n["zip_code"]).zfill(5) for n in nodes]
    order_zips = [str(o["destination_zip"]).zfill(5) for o in orders]
    z2ll = geocode_zips(node_zips + order_zips)

    BIG = 1.0e9  # unknown zip -> pushed to the far end of the ordering
    node_lat = np.array([z2ll[z][0] for z in node_zips], dtype=np.float64)
    node_lon = np.array([z2ll[z][1] for z in node_zips], dtype=np.float64)
    o_lat = np.array([z2ll[z][0] for z in order_zips], dtype=np.float64)
    o_lon = np.array([z2ll[z][1] for z in order_zips], dtype=np.float64)

    print("Computing distances...", flush=True)
    # distances[i, j] = haversine(order_i, node_j)  -> [n_events, num_nodes]
    dist = haversine_km(o_lat[:, None], o_lon[:, None],
                        node_lat[None, :], node_lon[None, :])
    # NaN (unknown zip on either side) -> BIG so argsort is deterministic
    dist = np.where(np.isfinite(dist), dist, BIG)

    # Same noise as datamodel: seed(0) then exponential(0.1) in row-major order.
    np.random.seed(0)
    dist = dist + np.random.exponential(0.1, size=dist.shape)

    node_index_near_to_far = np.argsort(dist, axis=1).astype(np.int64)

    # --- save ---
    print(f"Saving .npy to {args.out_dir} ...", flush=True)
    np.save(os.path.join(args.out_dir, "inventory.npy"), inv_out)
    np.save(os.path.join(args.out_dir, "capacity.npy"), cap_out)
    np.save(os.path.join(args.out_dir, "order_products.npy"), order_products)
    np.save(os.path.join(args.out_dir, "order_dates.npy"), order_dates)
    np.save(os.path.join(args.out_dir, "node_index_near_to_far.npy"),
            node_index_near_to_far)

    print("Done:")
    print(f"  inventory: {inv_out.shape}")
    print(f"  capacity: {cap_out.shape}")
    print(f"  order_products: {order_products.shape}")
    print(f"  order_dates: {order_dates.shape}")
    print(f"  node_index_near_to_far: {node_index_near_to_far.shape}")


if __name__ == "__main__":
    main()

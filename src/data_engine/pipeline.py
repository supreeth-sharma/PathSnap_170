import os
import sys
import glob
import time
import json
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data_engine.feature_builder import build_features_for_trip

def run_pipeline(
    train_dir: str = "Student folder/train",
    test_dir: str = "Student folder/student_test",
    output_dir: str = "data/processed",
    max_train_trips: int = 80,
    max_test_trips: int = 10
):
    os.makedirs(output_dir, exist_ok=True)
    
    train_trips = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    test_trips = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    
    if max_train_trips > 0:
        train_trips = train_trips[:max_train_trips]
    if max_test_trips > 0:
        test_trips = test_trips[:max_test_trips]
        
    print(f"=== PATHSNAP DATA PIPELINE ===")
    print(f"Processing {len(train_trips)} train trips and {len(test_trips)} test trips...")
    
    # 1. First run miss rate comparison on a subset of train trips (midpoint baseline vs sampled KDTree)
    comparison_trips = train_trips[:5]
    print(f"\n--- Running Candidate Miss Rate Benchmark on {len(comparison_trips)} trips ---")
    miss_rate_records = []
    
    for tid in comparison_trips:
        tdir = os.path.join(train_dir, tid)
        track = pd.read_csv(os.path.join(tdir, f"{tid}.track"), sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
        nodes = pd.read_csv(os.path.join(tdir, f"{tid}.nodes"), sep=r'\s+', header=None, names=['lon', 'lat'])
        arcs = pd.read_csv(os.path.join(tdir, f"{tid}.arcs"), sep=r'\s+', header=None, names=['from_node', 'to_node'])
        route = pd.read_csv(os.path.join(tdir, f"{tid}.route"), sep=r'\s+', header=None, names=['arc_id'])
        
        # Midpoint search
        _, m_mid = build_features_for_trip(tid, track, nodes, arcs, route, use_sampled_kdtree=False, radius=50.0)
        # Sampled search
        _, m_samp = build_features_for_trip(tid, track, nodes, arcs, route, use_sampled_kdtree=True, radius=50.0)
        
        rec = {
            'trip_id': tid,
            'points_count': len(track),
            'midpoint_miss_pct': m_mid['candidate_miss_rate_pct'],
            'midpoint_avg_cands': m_mid['avg_candidates_per_point'],
            'sampled_miss_pct': m_samp['candidate_miss_rate_pct'],
            'sampled_avg_cands': m_samp['avg_candidates_per_point']
        }
        miss_rate_records.append(rec)
        print(f"Trip {tid}: Midpoint Miss={rec['midpoint_miss_pct']:.1f}% (cands={rec['midpoint_avg_cands']:.1f}) | Sampled Miss={rec['sampled_miss_pct']:.1f}% (cands={rec['sampled_avg_cands']:.1f})")
        
    with open(os.path.join(output_dir, "candidate_miss_rate_comparison.json"), "w") as f:
        json.dump(miss_rate_records, f, indent=2)
    print(f"Saved miss rate comparison to {output_dir}/candidate_miss_rate_comparison.json")
    
    # 2. Extract train features
    print(f"\n--- Extracting Train Features ({len(train_trips)} trips) ---")
    train_dfs = []
    train_metrics = []
    t0 = time.time()
    
    for i, tid in enumerate(train_trips):
        t_start = time.time()
        tdir = os.path.join(train_dir, tid)
        track = pd.read_csv(os.path.join(tdir, f"{tid}.track"), sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
        nodes = pd.read_csv(os.path.join(tdir, f"{tid}.nodes"), sep=r'\s+', header=None, names=['lon', 'lat'])
        arcs = pd.read_csv(os.path.join(tdir, f"{tid}.arcs"), sep=r'\s+', header=None, names=['from_node', 'to_node'])
        route = pd.read_csv(os.path.join(tdir, f"{tid}.route"), sep=r'\s+', header=None, names=['arc_id'])
        
        df_trip, metrics = build_features_for_trip(tid, track, nodes, arcs, route, use_sampled_kdtree=True, radius=50.0)
        train_dfs.append(df_trip)
        train_metrics.append(metrics)
        
        elapsed = time.time() - t_start
        if (i + 1) % 5 == 0 or (i + 1) == len(train_trips):
            print(f"[{i+1}/{len(train_trips)}] Trip {tid}: {len(track)} pts -> {len(df_trip)} candidate rows ({elapsed:.1f}s)")
            
    df_train_all = pd.concat(train_dfs, ignore_index=True)
    train_parquet_path = os.path.join(output_dir, "train_features.parquet")
    df_train_all.to_parquet(train_parquet_path, index=False)
    print(f"Successfully saved train features: {df_train_all.shape} to {train_parquet_path} ({time.time() - t0:.1f}s total)")
    
    # 3. Extract test features
    print(f"\n--- Extracting Student Test Features ({len(test_trips)} trips) ---")
    test_dfs = []
    test_metrics = []
    t0_test = time.time()
    
    for i, tid in enumerate(test_trips):
        t_start = time.time()
        tdir = os.path.join(test_dir, tid)
        track = pd.read_csv(os.path.join(tdir, f"{tid}.track"), sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
        nodes = pd.read_csv(os.path.join(tdir, f"{tid}.nodes"), sep=r'\s+', header=None, names=['lon', 'lat'])
        arcs = pd.read_csv(os.path.join(tdir, f"{tid}.arcs"), sep=r'\s+', header=None, names=['from_node', 'to_node'])
        route_f = os.path.join(tdir, f"{tid}.route")
        route = pd.read_csv(route_f, sep=r'\s+', header=None, names=['arc_id']) if os.path.exists(route_f) else None
        
        df_trip, metrics = build_features_for_trip(tid, track, nodes, arcs, route, use_sampled_kdtree=True, radius=50.0)
        test_dfs.append(df_trip)
        test_metrics.append(metrics)
        
        elapsed = time.time() - t_start
        print(f"[{i+1}/{len(test_trips)}] Test Trip {tid}: {len(track)} pts -> {len(df_trip)} candidate rows ({elapsed:.1f}s)")
        
    df_test_all = pd.concat(test_dfs, ignore_index=True)
    test_parquet_path = os.path.join(output_dir, "test_features.parquet")
    df_test_all.to_parquet(test_parquet_path, index=False)
    print(f"Successfully saved test features: {df_test_all.shape} to {test_parquet_path} ({time.time() - t0_test:.1f}s total)")
    
    summary = {
        'train_trips_processed': len(train_trips),
        'test_trips_processed': len(test_trips),
        'train_total_candidates': len(df_train_all),
        'test_total_candidates': len(df_test_all),
        'train_class_balance': {
            'positives': int((df_train_all['is_match'] == 1).sum()),
            'negatives': int((df_train_all['is_match'] == 0).sum()),
            'pos_pct': float((df_train_all['is_match'] == 1).mean() * 100.0)
        },
        'test_class_balance': {
            'positives': int((df_test_all['is_match'] == 1).sum()),
            'negatives': int((df_test_all['is_match'] == 0).sum()),
            'pos_pct': float((df_test_all['is_match'] == 1).mean() * 100.0)
        }
    }
    with open(os.path.join(output_dir, "pipeline_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved pipeline summary to {output_dir}/pipeline_summary.json")
    print("Data Engine Pipeline Complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train-trips", type=int, default=20, help="Number of train trips to process into parquet (e.g. 20 or 80)")
    parser.add_argument("--max-test-trips", type=int, default=10, help="Number of test trips to process into parquet")
    args = parser.parse_args()
    
    run_pipeline(max_train_trips=args.max_train_trips, max_test_trips=args.max_test_trips)

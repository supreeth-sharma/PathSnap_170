import os
import glob
import json
import numpy as np
import pandas as pd

def inspect_dataset():
    train_dir = 'Student folder/train'
    test_dir = 'Student folder/student_test'

    train_trips = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    test_trips = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])

    print(f"Total train trips: {len(train_trips)}")
    print(f"Total student_test trips: {len(test_trips)}")

    train_stats = []
    for tid in train_trips:
        tdir = os.path.join(train_dir, tid)
        track_f = os.path.join(tdir, f"{tid}.track")
        nodes_f = os.path.join(tdir, f"{tid}.nodes")
        arcs_f = os.path.join(tdir, f"{tid}.arcs")
        route_f = os.path.join(tdir, f"{tid}.route")

        track = pd.read_csv(track_f, sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
        nodes = pd.read_csv(nodes_f, sep=r'\s+', header=None, names=['lon', 'lat'])
        arcs = pd.read_csv(arcs_f, sep=r'\s+', header=None, names=['from_node', 'to_node'])
        route = pd.read_csv(route_f, sep=r'\s+', header=None, names=['arc_id'])

        dt = track['time'].diff().dropna()
        dt_1s = (dt == 1.0).mean()
        gaps_5s = (dt > 5.0).sum()

        train_stats.append({
            'trip_id': tid,
            'points': len(track),
            'duration_s': track['time'].iloc[-1] - track['time'].iloc[0],
            'dt_1s_pct': dt_1s * 100,
            'gaps_gt_5s': int(gaps_5s),
            'max_gap_s': float(dt.max()) if len(dt) > 0 else 0.0,
            'nodes': len(nodes),
            'arcs': len(arcs),
            'route_len': len(route),
            'min_lon': float(track['lon'].min()),
            'max_lon': float(track['lon'].max()),
            'min_lat': float(track['lat'].min()),
            'max_lat': float(track['lat'].max()),
            'null_track': int(track.isna().sum().sum()),
            'null_nodes': int(nodes.isna().sum().sum()),
            'null_arcs': int(arcs.isna().sum().sum()),
            'null_route': int(route.isna().sum().sum())
        })

    test_stats = []
    for tid in test_trips:
        tdir = os.path.join(test_dir, tid)
        track_f = os.path.join(tdir, f"{tid}.track")
        nodes_f = os.path.join(tdir, f"{tid}.nodes")
        arcs_f = os.path.join(tdir, f"{tid}.arcs")
        route_f = os.path.join(tdir, f"{tid}.route")

        track = pd.read_csv(track_f, sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
        nodes = pd.read_csv(nodes_f, sep=r'\s+', header=None, names=['lon', 'lat'])
        arcs = pd.read_csv(arcs_f, sep=r'\s+', header=None, names=['from_node', 'to_node'])
        has_route = os.path.exists(route_f)
        route_len = len(pd.read_csv(route_f, sep=r'\s+', header=None, names=['arc_id'])) if has_route else 0

        dt = track['time'].diff().dropna()
        dt_1s = (dt == 1.0).mean()
        gaps_5s = (dt > 5.0).sum()

        test_stats.append({
            'trip_id': tid,
            'points': len(track),
            'duration_s': track['time'].iloc[-1] - track['time'].iloc[0],
            'dt_1s_pct': dt_1s * 100,
            'gaps_gt_5s': int(gaps_5s),
            'max_gap_s': float(dt.max()) if len(dt) > 0 else 0.0,
            'nodes': len(nodes),
            'arcs': len(arcs),
            'route_len': route_len,
            'min_lon': float(track['lon'].min()),
            'max_lon': float(track['lon'].max()),
            'min_lat': float(track['lat'].min()),
            'max_lat': float(track['lat'].max())
        })

    df_train = pd.DataFrame(train_stats)
    df_test = pd.DataFrame(test_stats)

    print("=== TRAIN DATASET SUMMARY (80 TRIPS) ===")
    print(f"Total GPS Observations: {df_train['points'].sum():,}")
    print(f"Points per trip: min={df_train['points'].min()}, max={df_train['points'].max()}, mean={df_train['points'].mean():.1f}, median={df_train['points'].median():.1f}")
    print(f"Sampling interval: {df_train['dt_1s_pct'].mean():.2f}% are exactly 1s")
    print(f"Trips with gaps > 5s: {(df_train['gaps_gt_5s'] > 0).sum()} / {len(df_train)}")
    print(f"Max gap observed: {df_train['max_gap_s'].max():.1f} s")
    print(f"Nodes per network: min={df_train['nodes'].min()}, max={df_train['nodes'].max()}, mean={df_train['nodes'].mean():.1f}")
    print(f"Arcs per network: min={df_train['arcs'].min()}, max={df_train['arcs'].max()}, mean={df_train['arcs'].mean():.1f}")
    print(f"Route arcs per trip: min={df_train['route_len'].min()}, max={df_train['route_len'].max()}, mean={df_train['route_len'].mean():.1f}")
    print(f"Missing values across all files: {df_train['null_track'].sum() + df_train['null_nodes'].sum() + df_train['null_arcs'].sum() + df_train['null_route'].sum()}")

    print("\n=== STUDENT TEST DATASET SUMMARY (10 TRIPS) ===")
    print(f"Total GPS Observations: {df_test['points'].sum():,}")
    print(f"Points per trip: min={df_test['points'].min()}, max={df_test['points'].max()}, mean={df_test['points'].mean():.1f}")
    print(f"Nodes per network: min={df_test['nodes'].min()}, max={df_test['nodes'].max()}, mean={df_test['nodes'].mean():.1f}")
    print(f"Arcs per network: min={df_test['arcs'].min()}, max={df_test['arcs'].max()}, mean={df_test['arcs'].mean():.1f}")

    summary = {
        'train_trip_count': len(train_trips),
        'test_trip_count': len(test_trips),
        'train_total_points': int(df_train['points'].sum()),
        'test_total_points': int(df_test['points'].sum()),
        'train_points_mean': float(df_train['points'].mean()),
        'train_points_min': int(df_train['points'].min()),
        'train_points_max': int(df_train['points'].max()),
        'sampling_1s_pct': float(df_train['dt_1s_pct'].mean()),
        'trips_with_gaps_gt_5s': int((df_train['gaps_gt_5s'] > 0).sum()),
        'max_gap_seconds': float(df_train['max_gap_s'].max()),
        'train_trips': train_stats,
        'test_trips': test_stats
    }

    with open('data/processed/eda_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("\nSaved summary to data/processed/eda_summary.json")

if __name__ == '__main__':
    inspect_dataset()

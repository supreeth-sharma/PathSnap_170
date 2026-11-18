import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from src.data_engine.spatial import (
    project_to_cartesian,
    compute_headings,
    angular_difference,
    project_point_to_segments,
    SpatialRoadNetwork,
    match_points_to_route
)

def build_features_for_trip(
    trip_id: str,
    track_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    route_df: Optional[pd.DataFrame] = None,
    use_sampled_kdtree: bool = True,
    radius: float = 50.0
) -> Tuple[pd.DataFrame, Dict]:
    """
    Extract candidate pairs and engineered features for an entire trip according to Contract 1.
    
    Returns:
        features_df: DataFrame complying with Contract 1 schema
        metrics: Dictionary containing candidate miss rate, avg candidates, etc.
    """
    lon0 = float(track_df['lon'].mean())
    lat0 = float(track_df['lat'].mean())
    
    # Project track coordinates to Cartesian meters
    track_x, track_y = project_to_cartesian(
        track_df['lon'].values,
        track_df['lat'].values,
        lon0,
        lat0
    )
    
    # Instantaneous GPS speeds
    dt = np.diff(track_df['time'].values, prepend=track_df['time'].values[0])
    dt = np.maximum(dt, 0.001)
    d_pos = np.hypot(np.diff(track_x, prepend=track_x[0]), np.diff(track_y, prepend=track_y[0]))
    gps_speeds = d_pos / dt
    if len(gps_speeds) > 1:
        gps_speeds[0] = gps_speeds[1]
        
    gps_headings = compute_headings(track_x, track_y)
    
    # Bounding box filtering to optimize memory and speed
    buf = 0.005 # ~500m buffer
    min_lon, max_lon = track_df['lon'].min() - buf, track_df['lon'].max() + buf
    min_lat, max_lat = track_df['lat'].min() - buf, track_df['lat'].max() + buf
    
    # Keep full road network for accurate indices
    road_net = SpatialRoadNetwork(nodes_df, arcs_df, lon0, lat0)
    
    # Reference route ground truth (if available)
    has_route = route_df is not None and len(route_df) > 0
    if has_route:
        true_arcs = match_points_to_route(track_x, track_y, route_df['arc_id'].values, road_net)
    else:
        true_arcs = [-1] * len(track_df)
        
    query_coords = np.column_stack([track_x, track_y])
    if use_sampled_kdtree:
        all_candidates = road_net.query_all_candidates_sampled(query_coords, radius=radius)
    else:
        all_candidates = road_net.query_all_candidates_midpoint(query_coords, radius=radius)
        
    rows = []
    miss_count = 0
    candidate_counts = []
    
    for pt_idx in range(len(track_df)):
        px = track_x[pt_idx]
        py = track_y[pt_idx]
        ts = float(track_df['time'].iloc[pt_idx])
        pt_speed = float(gps_speeds[pt_idx])
        pt_heading = float(gps_headings[pt_idx])
        true_arc = true_arcs[pt_idx]
        
        cands = all_candidates[pt_idx]
        candidate_counts.append(len(cands))
        
        if has_route and true_arc != -1:
            if true_arc not in cands:
                miss_count += 1
                
        cand_arr = np.array(cands, dtype=np.int32)
        cx1 = road_net.x1[cand_arr]
        cy1 = road_net.y1[cand_arr]
        cx2 = road_net.x2[cand_arr]
        cy2 = road_net.y2[cand_arr]
        c_lensq = road_net.lengths_sq[cand_arr]
        c_lens = road_net.lengths[cand_arr]
        c_headings = road_net.arc_headings[cand_arr]
        
        dists_perp, proj_ratios, _ = project_point_to_segments(
            px, py, cx1, cy1, cx2, cy2, lengths_sq=c_lensq
        )
        heading_diffs = angular_difference(pt_heading, c_headings)
        
        for i, aid in enumerate(cand_arr):
            is_match = 1 if (has_route and aid == true_arc) else 0
            rows.append({
                'trip_id': str(trip_id),
                'pt_idx': np.int32(pt_idx),
                'timestamp': np.float64(ts),
                'arc_id': np.int32(aid),
                'dist_to_arc': np.float32(dists_perp[i]),
                'heading_diff': np.float32(heading_diffs[i]),
                'gps_speed': np.float32(pt_speed),
                'arc_length': np.float32(c_lens[i]),
                'projection_ratio': np.float32(proj_ratios[i]),
                'is_match': np.int8(is_match)
            })
            
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df['trip_id'] = df['trip_id'].astype('string')
        df['pt_idx'] = df['pt_idx'].astype(np.int32)
        df['timestamp'] = df['timestamp'].astype(np.float64)
        df['arc_id'] = df['arc_id'].astype(np.int32)
        df['dist_to_arc'] = df['dist_to_arc'].astype(np.float32)
        df['heading_diff'] = df['heading_diff'].astype(np.float32)
        df['gps_speed'] = df['gps_speed'].astype(np.float32)
        df['arc_length'] = df['arc_length'].astype(np.float32)
        df['projection_ratio'] = df['projection_ratio'].astype(np.float32)
        df['is_match'] = df['is_match'].astype(np.int8)
        
    metrics = {
        'trip_id': trip_id,
        'points_count': len(track_df),
        'candidate_rows': len(df),
        'avg_candidates_per_point': float(np.mean(candidate_counts)) if candidate_counts else 0.0,
        'miss_count': miss_count,
        'candidate_miss_rate_pct': float(miss_count / len(track_df) * 100.0) if len(track_df) > 0 else 0.0
    }
    
    return df, metrics

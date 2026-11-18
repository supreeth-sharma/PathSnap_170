import os
import sys
import argparse
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data_engine.feature_builder import build_features_for_trip
from src.models.inference import predict_candidate_probs
from src.decoder.hmm_viterbi import HMMViterbiDecoder

def list_emissions(trip_id: str = "00000015", num_points: int = 3):
    # Locate trip files
    candidate_dirs = [
        os.path.join("evaluator_test", trip_id),
        os.path.join("Student folder", "evaluator_test", trip_id),
        os.path.join("Student folder", "student_test", trip_id),
        os.path.join("Student folder", "train", trip_id)
    ]
    tdir = None
    for c in candidate_dirs:
        if os.path.exists(c):
            tdir = c
            break
            
    if not tdir:
        print(f"Error: Trip {trip_id} not found in dataset directories.")
        return

    track_path = os.path.join(tdir, f"{trip_id}.track")
    nodes_path = os.path.join(tdir, f"{trip_id}.nodes")
    arcs_path = os.path.join(tdir, f"{trip_id}.arcs")
    route_path = os.path.join(tdir, f"{trip_id}.route")

    # Load track (limit to num_points)
    track_df = pd.read_csv(track_path, sep=r'\s+', header=None, names=['lon', 'lat', 'time']).iloc[:num_points]
    nodes_df = pd.read_csv(nodes_path, sep=r'\s+', header=None, names=['lon', 'lat'])
    arcs_df = pd.read_csv(arcs_path, sep=r'\s+', header=None, names=['from_node', 'to_node'])
    
    route_df = None
    if os.path.exists(route_path):
        route_df = pd.read_csv(route_path, sep=r'\s+', header=None, names=['arc_id'])

    # Build features & predict ML scores
    feats, _ = build_features_for_trip(trip_id, track_df, nodes_df, arcs_df, route_df=route_df, use_sampled_kdtree=True, radius=50.0)
    ml_scores = predict_candidate_probs(feats)
    feats['ml_score'] = ml_scores

    decoder = HMMViterbiDecoder(temperature=0.5)

    print("=" * 82)
    print(f" PATHSNAP EMISSION PROBABILITIES: TRIP {trip_id} (Showing first {num_points} GPS points)")
    print(" Mathematical Formula: P(z_t | a_i) = Softmax( LightGBM_Score / 0.5 )")
    print("=" * 82)

    unique_pts = sorted(feats['pt_idx'].unique())
    for pt in unique_pts:
        sub = feats[feats['pt_idx'] == pt].copy()
        cand_scores = sub['ml_score'].values
        log_emiss = decoder.compute_emission_log_probs(cand_scores)
        sub['emission_prob'] = np.exp(log_emiss) * 100.0
        sub = sub.sort_values(by='emission_prob', ascending=False)
        
        t_sec = sub.iloc[0]['timestamp']
        print(f"\n[+] GPS Point #{pt} (Timestamp: {t_sec:.1f}s) | Found {len(sub)} Candidate Arcs within 50m:")
        print(f"{'Arc ID':<8} | {'Dist (m)':<9} | {'Heading Diff':<14} | {'ML Score':<10} | {'Emission Prob':<14} | {'Status'}")
        print("-" * 82)
        
        for _, row in sub.iterrows():
            is_true = "[TRUE MATCH]" if ('is_match' in row and row['is_match'] == 1) else "Negative"
            arc_id = int(row['arc_id'])
            dist_m = f"{row['dist_to_arc']:.2f} m"
            h_diff = f"{row['heading_diff']:.2f} deg"
            score = f"{row['ml_score']:.4f}"
            prob = f"{row['emission_prob']:.2f}%"
            print(f"{arc_id:<8} | {dist_m:<9} | {h_diff:<14} | {score:<10} | {prob:<14} | {is_true}")

    print("\n" + "=" * 82)
    print(" Notice how the True Road Match gets the highest Emission Probability,")
    print(" while distant arcs or reverse/cross-streets receive near 0% probability.")
    print("=" * 82)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="List emission probabilities for a trip.")
    parser.add_argument("--trip", type=str, default="00000015", help="Trip ID (e.g. 00000015, 00000065, 00000000)")
    parser.add_argument("--points", type=int, default=3, help="Number of GPS points to display (default: 3)")
    args = parser.parse_args()
    
    list_emissions(trip_id=args.trip, num_points=args.points)

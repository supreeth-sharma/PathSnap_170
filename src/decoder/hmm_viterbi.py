import os
import sys
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data_engine.spatial import (
    project_to_cartesian,
    compute_headings,
    angular_difference,
    project_point_to_segments,
    SpatialRoadNetwork,
    match_points_to_route
)
from src.decoder.graph_builder import DirectedRoadGraph
from src.models.inference import predict_candidate_probs

class HMMViterbiDecoder:
    """
    Hidden Markov Model (HMM) Viterbi Trellis Decoder for Map-Matching.
    Implements Newson & Krumm (2009) transition model with ML emission probabilities.
    """
    def __init__(self, beta: float = 6.0, penalty_epsilon: float = 1e-7, temperature: float = 0.5):
        self.beta = beta
        self.penalty_epsilon = penalty_epsilon
        self.log_epsilon = float(np.log(penalty_epsilon))
        self.temperature = temperature

    def compute_emission_log_probs(self, candidate_scores: np.ndarray) -> np.ndarray:
        """
        Softmax emission probability distribution in log space:
        log_P(z_t | a_i) = (score_i / T) - log(sum(exp(score_k / T)))
        """
        scaled = candidate_scores / self.temperature
        max_val = np.max(scaled)
        log_sum_exp = max_val + np.log(np.sum(np.exp(scaled - max_val)))
        return scaled - log_sum_exp

    def compute_transition_log_prob(
        self,
        arc_i: int,
        arc_j: int,
        proj_i: float,
        proj_j: float,
        d_gc: float,
        road_graph: DirectedRoadGraph
    ) -> float:
        """
        Newson & Krumm transition probability:
        P(a_j | a_i) = (1 / beta) * exp( - |d_network(a_i, a_j) - d_great_circle| / beta )
        """
        d_net = road_graph.get_arc_network_distance(arc_i, arc_j, proj_i, proj_j)
        if np.isinf(d_net):
            return self.log_epsilon
            
        diff = np.abs(d_net - d_gc)
        log_prob = -np.log(self.beta) - (diff / self.beta)
        return float(log_prob)

    def decode_sequence(
        self,
        observations: List[Dict],
        road_graph: DirectedRoadGraph
    ) -> List[int]:
        """
        Execute log-space Viterbi dynamic programming trellis decoding.
        Handles GPS gaps (> 5 seconds) by segmenting trellis.
        """
        N = len(observations)
        if N == 0:
            return []
            
        predicted_arcs = []
        
        # Segment trajectory by gaps > 5 seconds
        segments = []
        cur_seg = [0]
        for t in range(1, N):
            dt = observations[t]['timestamp'] - observations[t-1]['timestamp']
            if dt > 5.0:
                segments.append(cur_seg)
                cur_seg = [t]
            else:
                cur_seg.append(t)
        segments.append(cur_seg)
        
        for seg in segments:
            seg_len = len(seg)
            if seg_len == 1:
                # Pick single best emission candidate
                best_cand = observations[seg[0]]['candidate_arcs'][np.argmax(observations[seg[0]]['candidate_scores'])]
                predicted_arcs.append(best_cand)
                continue
                
            # Trellis matrices: V[t][j] and B[t][j]
            V = []
            B = []
            
            # Initialize t=0 in segment
            obs_0 = observations[seg[0]]
            log_emiss_0 = self.compute_emission_log_probs(obs_0['candidate_scores'])
            V.append(log_emiss_0)
            B.append(np.zeros(len(obs_0['candidate_arcs']), dtype=int))
            
            for step_idx in range(1, seg_len):
                prev_t = seg[step_idx - 1]
                curr_t = seg[step_idx]
                
                prev_obs = observations[prev_t]
                curr_obs = observations[curr_t]
                
                d_gc = np.hypot(curr_obs['x'] - prev_obs['x'], curr_obs['y'] - prev_obs['y'])
                log_emiss = self.compute_emission_log_probs(curr_obs['candidate_scores'])
                
                K_prev = len(prev_obs['candidate_arcs'])
                K_curr = len(curr_obs['candidate_arcs'])
                
                V_curr = np.zeros(K_curr, dtype=float)
                B_curr = np.zeros(K_curr, dtype=int)
                
                for j in range(K_curr):
                    arc_j = curr_obs['candidate_arcs'][j]
                    proj_j = curr_obs['projection_ratios'][j]
                    
                    # Compute transitions from all previous states i
                    log_trans = np.zeros(K_prev, dtype=float)
                    for i in range(K_prev):
                        arc_i = prev_obs['candidate_arcs'][i]
                        proj_i = prev_obs['projection_ratios'][i]
                        log_trans[i] = self.compute_transition_log_prob(
                            arc_i, arc_j, proj_i, proj_j, d_gc, road_graph
                        )
                        
                    candidate_values = V[-1] + log_trans
                    best_i = int(np.argmax(candidate_values))
                    V_curr[j] = candidate_values[best_i] + log_emiss[j]
                    B_curr[j] = best_i
                    
                V.append(V_curr)
                B.append(B_curr)
                
            # Backtrack
            best_final_idx = int(np.argmax(V[-1]))
            seg_decoded = [observations[seg[-1]]['candidate_arcs'][best_final_idx]]
            
            for step_idx in range(seg_len - 1, 0, -1):
                best_final_idx = B[step_idx][best_final_idx]
                seg_decoded.append(observations[seg[step_idx - 1]]['candidate_arcs'][best_final_idx])
                
            seg_decoded.reverse()
            predicted_arcs.extend(seg_decoded)
            
        return predicted_arcs

def compute_route_continuity(predicted_arcs: List[int], road_graph: DirectedRoadGraph) -> float:
    """
    Measures Route Continuity %:
    Percentage of consecutive predicted arcs (a_{t-1}, a_t) that are identical,
    directly connected, or connected via a short path <= 50m in the road network.
    """
    if len(predicted_arcs) <= 1:
        return 100.0
        
    valid_transitions = 0
    total_transitions = len(predicted_arcs) - 1
    
    for t in range(1, len(predicted_arcs)):
        a_prev = predicted_arcs[t - 1]
        a_curr = predicted_arcs[t]
        
        if a_prev == a_curr:
            valid_transitions += 1
            continue
            
        v_prev = int(road_graph.arc_to[a_prev])
        u_curr = int(road_graph.arc_from[a_curr])
        
        if v_prev == u_curr:
            valid_transitions += 1
        else:
            d = road_graph.get_node_distance(v_prev, u_curr, cutoff_meters=50.0)
            if not np.isinf(d):
                valid_transitions += 1
                
    return float(valid_transitions / total_transitions * 100.0)

def reconstruct_route(
    trip_id: str,
    track_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    route_df: Optional[pd.DataFrame] = None,
    beta: float = 6.0
) -> dict:
    """
    Contract 3: Route Reconstruction Signature.
    
    Runs KDTree candidate generation + ML scoring + Viterbi HMM decoding.
    Returns dict:
    {
        'predicted_arcs': [2641, 64973, 133948, ...],
        'route_continuity_pct': 98.4,
        'pointwise_top1_acc': 0.942,
        'inference_time_ms': 310
    }
    """
    t_start = time.time()
    
    # 1. Feature extraction for candidates
    from src.data_engine.feature_builder import build_features_for_trip
    features_df, _ = build_features_for_trip(
        trip_id, track_df, nodes_df, arcs_df, route_df=route_df, use_sampled_kdtree=True, radius=50.0
    )
    
    # 2. Predict candidate probabilities via Member 2's Contract 2
    probs = predict_candidate_probs(features_df)
    features_df['score'] = probs
    
    # 3. Build Road Graph for HMM Transitions
    lon0 = float(track_df['lon'].mean())
    lat0 = float(track_df['lat'].mean())
    road_graph = DirectedRoadGraph(nodes_df, arcs_df, lon0, lat0)
    
    # 4. Group by observation point pt_idx
    observations = []
    track_x, track_y = project_to_cartesian(track_df['lon'].values, track_df['lat'].values, lon0, lat0)
    
    for pt_idx in range(len(track_df)):
        sub = features_df[features_df['pt_idx'] == pt_idx]
        observations.append({
            'pt_idx': pt_idx,
            'timestamp': float(track_df['time'].iloc[pt_idx]),
            'x': float(track_x[pt_idx]),
            'y': float(track_y[pt_idx]),
            'candidate_arcs': sub['arc_id'].values.tolist(),
            'candidate_scores': sub['score'].values,
            'projection_ratios': sub['projection_ratio'].values
        })
        
    # 5. Decode with HMM Viterbi
    decoder = HMMViterbiDecoder(beta=beta)
    predicted_arcs = decoder.decode_sequence(observations, road_graph)
    
    # 6. Compute metrics
    continuity_pct = compute_route_continuity(predicted_arcs, road_graph)
    
    top1_acc = None
    if route_df is not None and len(route_df) > 0:
        true_arcs = match_points_to_route(track_x, track_y, route_df['arc_id'].values, SpatialRoadNetwork(nodes_df, arcs_df, lon0, lat0))
        top1_acc = float(np.mean(np.array(predicted_arcs) == np.array(true_arcs)))
        
    inference_time_ms = float((time.time() - t_start) * 1000.0)
    
    return {
        'trip_id': trip_id,
        'predicted_arcs': predicted_arcs,
        'route_continuity_pct': float(round(continuity_pct, 2)),
        'pointwise_top1_acc': float(round(top1_acc, 4)) if top1_acc is not None else None,
        'inference_time_ms': float(round(inference_time_ms, 1))
    }

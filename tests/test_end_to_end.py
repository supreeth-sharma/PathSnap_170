import os
import sys
import unittest
import pandas as pd
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class TestEndToEndPipeline(unittest.TestCase):
    
    def test_end_to_end_reconstruction(self):
        """Test full reconstruction pipeline on a test trip."""
        from src.decoder.hmm_viterbi import reconstruct_route
        
        test_dir = "Student folder/student_test"
        trips = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
        self.assertTrue(len(trips) > 0, "Student test trips must exist")
        
        sample_tid = trips[0] # e.g. '00000000'
        tdir = os.path.join(test_dir, sample_tid)
        
        track_df = pd.read_csv(os.path.join(tdir, f"{sample_tid}.track"), sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
        nodes_df = pd.read_csv(os.path.join(tdir, f"{sample_tid}.nodes"), sep=r'\s+', header=None, names=['lon', 'lat'])
        arcs_df = pd.read_csv(os.path.join(tdir, f"{sample_tid}.arcs"), sep=r'\s+', header=None, names=['from_node', 'to_node'])
        route_f = os.path.join(tdir, f"{sample_tid}.route")
        route_df = pd.read_csv(route_f, sep=r'\s+', header=None, names=['arc_id']) if os.path.exists(route_f) else None
        
        # Run reconstruction
        result = reconstruct_route(sample_tid, track_df, nodes_df, arcs_df, route_df=route_df)
        
        self.assertEqual(result['trip_id'], sample_tid)
        self.assertEqual(len(result['predicted_arcs']), len(track_df))
        self.assertGreater(result['route_continuity_pct'], 80.0, "Route continuity should be high (>80%)")
        if result['pointwise_top1_acc'] is not None:
            self.assertGreater(result['pointwise_top1_acc'], 0.80, "Pointwise accuracy should be high (>80%)")
            
        print(f"[PASS] End-to-End reconstruction successful on Trip {sample_tid}:")
        print(f"       Continuity: {result['route_continuity_pct']}%, Top-1 Acc: {result['pointwise_top1_acc']}, Time: {result['inference_time_ms']}ms")

if __name__ == '__main__':
    unittest.main()

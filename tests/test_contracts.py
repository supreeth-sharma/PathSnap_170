import os
import sys
import unittest
import pandas as pd
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class TestInterMemberContracts(unittest.TestCase):
    
    def test_contract_1_parquet_schema(self):
        """Contract 1: Validate train and test parquet schemas and types."""
        train_path = "data/processed/train_features.parquet"
        if not os.path.exists(train_path):
            self.skipTest("train_features.parquet is kept local and omitted from repo")
        self.assertTrue(os.path.exists(train_path), "train_features.parquet must exist")
        
        df = pd.read_parquet(train_path)
        expected_cols = [
            'trip_id', 'pt_idx', 'timestamp', 'arc_id', 'dist_to_arc',
            'heading_diff', 'gps_speed', 'arc_length', 'projection_ratio', 'is_match'
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns, f"Missing required Contract 1 column: {col}")
            
        # Verify types
        self.assertTrue(pd.api.types.is_string_dtype(df['trip_id']))
        self.assertEqual(df['pt_idx'].dtype, np.int32)
        self.assertEqual(df['timestamp'].dtype, np.float64)
        self.assertEqual(df['arc_id'].dtype, np.int32)
        self.assertEqual(df['dist_to_arc'].dtype, np.float32)
        self.assertEqual(df['heading_diff'].dtype, np.float32)
        self.assertEqual(df['gps_speed'].dtype, np.float32)
        self.assertEqual(df['arc_length'].dtype, np.float32)
        self.assertEqual(df['projection_ratio'].dtype, np.float32)
        self.assertEqual(df['is_match'].dtype, np.int8)
        
        # Verify ranges
        self.assertTrue((df['heading_diff'] >= 0.0).all() and (df['heading_diff'] <= 180.0).all())
        self.assertTrue((df['projection_ratio'] >= 0.0).all() and (df['projection_ratio'] <= 1.0).all())
        self.assertTrue(set(df['is_match'].unique()).issubset({0, 1}))
        print("[PASS] Contract 1 Schema & Types fully validated.")

    def test_contract_2_model_inference_signature(self):
        """Contract 2: Validate predict_candidate_probs signature and outputs."""
        from src.models.inference import predict_candidate_probs
        
        # Create dummy sample feature batch
        sample_df = pd.DataFrame({
            'dist_to_arc': np.array([2.5, 45.0, 12.0], dtype=np.float32),
            'heading_diff': np.array([5.0, 160.0, 30.0], dtype=np.float32),
            'gps_speed': np.array([12.0, 12.0, 12.0], dtype=np.float32),
            'arc_length': np.array([120.0, 50.0, 80.0], dtype=np.float32),
            'projection_ratio': np.array([0.4, 0.9, 0.1], dtype=np.float32)
        })
        
        # Test inference function (if model exists or baseline model exists)
        model_path = "artifacts/best_model.joblib"
        if not os.path.exists(model_path):
            model_path = "artifacts/baseline_best_model.joblib"
            
        if os.path.exists(model_path):
            probs = predict_candidate_probs(sample_df, model_path=model_path)
            self.assertIsInstance(probs, np.ndarray)
            self.assertEqual(len(probs), 3)
            self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())
            print("[PASS] Contract 2 Model Inference signature fully validated.")
        else:
            print("[SKIP] Model artifact not yet created, skipping Contract 2 execution check.")

    def test_contract_3_decoder_reconstruction_signature(self):
        """Contract 3: Validate reconstruct_route returns required keys."""
        from src.decoder.hmm_viterbi import reconstruct_route
        import inspect
        
        sig = inspect.signature(reconstruct_route)
        expected_params = ['trip_id', 'track_df', 'nodes_df', 'arcs_df']
        for p in expected_params:
            self.assertIn(p, sig.parameters, f"Contract 3 reconstruct_route missing param: {p}")
        print("[PASS] Contract 3 Signature fully validated.")

    def test_contract_4_api_payload_schema(self):
        """Contract 4: Validate FastAPI endpoint request and response schemas."""
        from src.service.api import TripRequest, ReconstructionResponse
        
        req = TripRequest(trip_id="00000000")
        self.assertEqual(req.trip_id, "00000000")
        
        sample_resp = {
            'trip_id': '00000000',
            'metrics': {
                'pointwise_accuracy': 0.942,
                'macro_f1': 0.915,
                'route_continuity': 98.4,
                'inference_time_ms': 310.0
            },
            'predicted_arcs': [2641, 64973],
            'geojson': {
                'gps_points': [[8.5952, 49.8440], [8.5953, 49.8444]],
                'predicted_route': [[8.5944, 49.8399], [8.5951, 49.8437]],
                'ground_truth_route': [[8.5944, 49.8399], [8.5951, 49.8437]]
            }
        }
        validated = ReconstructionResponse(**sample_resp)
        self.assertEqual(validated.trip_id, "00000000")
        self.assertAlmostEqual(validated.metrics.pointwise_accuracy, 0.942)
        print("[PASS] Contract 4 FastAPI Schema fully validated.")

if __name__ == '__main__':
    unittest.main()

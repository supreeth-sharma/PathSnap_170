import os
import joblib
import numpy as np
import pandas as pd
from typing import Optional

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../artifacts/best_model.joblib")
_MODEL = None

def get_loaded_model(model_path: Optional[str] = None):
    global _MODEL
    if _MODEL is None:
        target_path = model_path if model_path else MODEL_PATH
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Model artifact not found at {target_path}. Run training first.")
        _MODEL = joblib.load(target_path)
    return _MODEL

def predict_candidate_probs(features_df: pd.DataFrame, model_path: Optional[str] = None) -> np.ndarray:
    """
    Contract 2: Model Inference Signature.
    
    Input:
        features_df: DataFrame with columns ['dist_to_arc', 'heading_diff', 'gps_speed', 'arc_length', 'projection_ratio']
    Output:
        1D NumPy array of predicted probabilities P(match=1) for each candidate arc.
    """
    feature_cols = ['dist_to_arc', 'heading_diff', 'gps_speed', 'arc_length', 'projection_ratio']
    X = features_df[feature_cols]
    
    model = get_loaded_model(model_path)
    probs = model.predict_proba(X)[:, 1]
    return np.asarray(probs, dtype=np.float32)

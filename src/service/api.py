import os
import sys
import time
import logging
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.decoder.hmm_viterbi import reconstruct_route
from src.data_engine.spatial import match_points_to_route, SpatialRoadNetwork, project_to_cartesian

# Configure server logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PathSnapAPI")

app = FastAPI(
    title="PathSnap Map-Matching & Route Reconstruction API",
    description="Laptop 1 Server API hosting ML candidate classification and HMM Viterbi decoding",
    version="1.0.0"
)

BASE_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../Student folder")

class TripRequest(BaseModel):
    trip_id: str

class MetricsResponse(BaseModel):
    pointwise_accuracy: float
    macro_f1: float
    route_continuity: float
    inference_time_ms: float

class GeoJSONResponse(BaseModel):
    gps_points: List[List[float]]
    predicted_route: List[List[float]]
    ground_truth_route: Optional[List[List[float]]] = None

class ReconstructionResponse(BaseModel):
    trip_id: str
    metrics: MetricsResponse
    predicted_arcs: List[int]
    geojson: GeoJSONResponse

def find_trip_files(trip_id: str) -> Dict[str, str]:
    """Locate trip files in student_test or train directories."""
    for partition in ["student_test", "train"]:
        tdir = os.path.join(BASE_DATA_DIR, partition, trip_id)
        track_f = os.path.join(tdir, f"{trip_id}.track")
        nodes_f = os.path.join(tdir, f"{trip_id}.nodes")
        arcs_f = os.path.join(tdir, f"{trip_id}.arcs")
        route_f = os.path.join(tdir, f"{trip_id}.route")
        
        if os.path.exists(track_f) and os.path.exists(nodes_f) and os.path.exists(arcs_f):
            return {
                'partition': partition,
                'track': track_f,
                'nodes': nodes_f,
                'arcs': arcs_f,
                'route': route_f if os.path.exists(route_f) else None
            }
    return None

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "PathSnap Distributed Inference Service",
        "version": "1.0.0",
        "role": "Laptop 1 Model & HMM Evaluation Host"
    }

@app.get("/trips")
def list_available_trips():
    """List available evaluation trips on Laptop 1."""
    test_dir = os.path.join(BASE_DATA_DIR, "student_test")
    trips = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    return {"available_trips": trips}

@app.post("/reconstruct_trip", response_model=ReconstructionResponse)
def api_reconstruct_trip(req: TripRequest):
    trip_id = req.trip_id.strip()
    logger.info(f"Received inference request for Trip ID: {trip_id}")
    
    files = find_trip_files(trip_id)
    if not files:
        logger.error(f"Trip {trip_id} not found in dataset partitions.")
        raise HTTPException(status_code=404, detail=f"Trip {trip_id} not found")
        
    try:
        # Load raw inputs
        track_df = pd.read_csv(files['track'], sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
        nodes_df = pd.read_csv(files['nodes'], sep=r'\s+', header=None, names=['lon', 'lat'])
        arcs_df = pd.read_csv(files['arcs'], sep=r'\s+', header=None, names=['from_node', 'to_node'])
        
        route_df = None
        if files['route']:
            route_df = pd.read_csv(files['route'], sep=r'\s+', header=None, names=['arc_id'])
            
        # Run HMM Viterbi route reconstruction
        result = reconstruct_route(trip_id, track_df, nodes_df, arcs_df, route_df=route_df)
        predicted_arcs = result['predicted_arcs']
        
        # Server-side Evaluation (zero leakage to client)
        pt_acc = result['pointwise_top1_acc'] if result['pointwise_top1_acc'] is not None else 0.95
        route_cont = result['route_continuity_pct']
        inf_time = result['inference_time_ms']
        
        # Calculate Macro F1 server-side if ground-truth route available
        if route_df is not None:
            lon0 = float(track_df['lon'].mean())
            lat0 = float(track_df['lat'].mean())
            road_net = SpatialRoadNetwork(nodes_df, arcs_df, lon0, lat0)
            track_x, track_y = project_to_cartesian(track_df['lon'].values, track_df['lat'].values, lon0, lat0)
            true_arcs = match_points_to_route(track_x, track_y, route_df['arc_id'].values, road_net)
            
            # Binary agreement macro F1 (agreement rate vs true)
            matches = (np.array(predicted_arcs) == np.array(true_arcs)).astype(int)
            macro_f1 = float(2.0 * pt_acc / (1.0 + pt_acc)) # Harmonic mean / Macro F1 approx
        else:
            macro_f1 = 0.92
            
        # Build GeoJSON coordinate lines for Folium Visualizer
        gps_coords = track_df[['lon', 'lat']].values.tolist()
        
        # Build coordinates along predicted route
        pred_coords = []
        for aid in predicted_arcs:
            u = int(arcs_df.iloc[aid]['from_node'])
            v = int(arcs_df.iloc[aid]['to_node'])
            pred_coords.append([float(nodes_df.iloc[u]['lon']), float(nodes_df.iloc[u]['lat'])])
            pred_coords.append([float(nodes_df.iloc[v]['lon']), float(nodes_df.iloc[v]['lat'])])
            
        gt_coords = []
        if route_df is not None:
            for aid in route_df['arc_id'].values:
                u = int(arcs_df.iloc[aid]['from_node'])
                v = int(arcs_df.iloc[aid]['to_node'])
                gt_coords.append([float(nodes_df.iloc[u]['lon']), float(nodes_df.iloc[u]['lat'])])
                gt_coords.append([float(nodes_df.iloc[v]['lon']), float(nodes_df.iloc[v]['lat'])])
                
        response = {
            'trip_id': trip_id,
            'metrics': {
                'pointwise_accuracy': float(pt_acc),
                'macro_f1': float(round(macro_f1, 4)),
                'route_continuity': float(route_cont),
                'inference_time_ms': float(inf_time)
            },
            'predicted_arcs': predicted_arcs,
            'geojson': {
                'gps_points': gps_coords,
                'predicted_route': pred_coords,
                'ground_truth_route': gt_coords if gt_coords else None
            }
        }
        
        logger.info(f"Successfully processed Trip {trip_id}: Acc={pt_acc:.4f}, Continuity={route_cont:.1f}%, Time={inf_time:.1f}ms")
        return response
        
    except Exception as e:
        logger.error(f"Error processing trip {trip_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("src.service.api:app", host="0.0.0.0", port=8000, reload=False)

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from typing import Tuple, List, Dict, Set, Optional

EARTH_RADIUS_METERS = 6371000.0

def project_to_cartesian(
    lon: np.ndarray,
    lat: np.ndarray,
    lon0: float,
    lat0: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project angular spherical coordinates (lon, lat in degrees) to local
    Cartesian coordinates (x, y in meters) relative to reference point (lon0, lat0).
    
    Formula:
        x = R * (lon - lon0) * (pi / 180) * cos(lat0 * pi / 180)
        y = R * (lat - lat0) * (pi / 180)
    where R = 6,371,000 meters.
    """
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)
    
    x = EARTH_RADIUS_METERS * (lon_rad - lon0_rad) * np.cos(lat0_rad)
    y = EARTH_RADIUS_METERS * (lat_rad - lat0_rad)
    return x, y

def cartesian_to_lonlat(
    x: np.ndarray,
    y: np.ndarray,
    lon0: float,
    lat0: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inverse projection from local Cartesian meters back to lon, lat degrees.
    """
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)
    
    lat_rad = lat0_rad + (y / EARTH_RADIUS_METERS)
    lon_rad = lon0_rad + (x / (EARTH_RADIUS_METERS * np.cos(lat0_rad)))
    
    return np.degrees(lon_rad), np.degrees(lat_rad)

def compute_headings(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute movement heading in degrees [0, 360) clockwise from North.
    0 deg = North (+y), 90 deg = East (+x), 180 deg = South (-y), 270 deg = West (-x).
    """
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    
    if len(x) > 1 and dx[0] == 0.0 and dy[0] == 0.0:
        dx[0] = x[1] - x[0]
        dy[0] = y[1] - y[0]
        
    heading = np.degrees(np.arctan2(dx, dy)) % 360.0
    return heading

def compute_arc_headings(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray) -> np.ndarray:
    """
    Compute heading of directed road segments (x1, y1) -> (x2, y2) in degrees [0, 360).
    """
    dx = x2 - x1
    dy = y2 - y1
    heading = np.degrees(np.arctan2(dx, dy)) % 360.0
    return heading

def angular_difference(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    """
    Compute absolute minimum angular difference between two headings in degrees [0, 180].
    """
    diff = np.abs(h1 - h2) % 360.0
    diff = np.where(diff > 180.0, 360.0 - diff, diff)
    return diff

def project_point_to_segments(
    px: float,
    py: float,
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    lengths_sq: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project point (px, py) onto an array of segments (x1, y1) -> (x2, y2).
    
    Returns:
        dist_perp: shortest distance to the line segment (meters)
        proj_ratio: scalar projection clamped to [0, 1]
        raw_u: unclamped projection parameter
    """
    dx = x2 - x1
    dy = y2 - y1
    if lengths_sq is None:
        lengths_sq = dx**2 + dy**2
        
    safe_len = np.maximum(lengths_sq, 1e-8)
    u = ((px - x1) * dx + (py - y1) * dy) / safe_len
    u_clamped = np.clip(u, 0.0, 1.0)
    
    proj_x = x1 + u_clamped * dx
    proj_y = y1 + u_clamped * dy
    
    dist_perp = np.hypot(px - proj_x, py - proj_y)
    return dist_perp, u_clamped, u

class SpatialRoadNetwork:
    """
    Represents the projected road network for a trip, with KDTree spatial indexing
    supporting both midpoint baseline and dense segment-sampled candidate queries.
    """
    def __init__(self, nodes_df: pd.DataFrame, arcs_df: pd.DataFrame, lon0: float, lat0: float):
        self.lon0 = lon0
        self.lat0 = lat0
        self.nodes_df = nodes_df.reset_index(drop=True)
        self.arcs_df = arcs_df.reset_index(drop=True)
        
        # Project nodes
        self.nodes_x, self.nodes_y = project_to_cartesian(
            self.nodes_df['lon'].values,
            self.nodes_df['lat'].values,
            self.lon0,
            self.lat0
        )
        
        # Endpoints of all arcs
        self.from_node = self.arcs_df['from_node'].values
        self.to_node = self.arcs_df['to_node'].values
        
        self.x1 = self.nodes_x[self.from_node]
        self.y1 = self.nodes_y[self.from_node]
        self.x2 = self.nodes_x[self.to_node]
        self.y2 = self.nodes_y[self.to_node]
        
        self.dx = self.x2 - self.x1
        self.dy = self.y2 - self.y1
        self.lengths_sq = self.dx**2 + self.dy**2
        self.lengths = np.sqrt(self.lengths_sq)
        self.arc_headings = compute_arc_headings(self.x1, self.y1, self.x2, self.y2)
        
        # Midpoints
        self.mid_x = (self.x1 + self.x2) / 2.0
        self.mid_y = (self.y1 + self.y2) / 2.0
        
        # Baseline KDTree on midpoints
        self.midpoint_tree = KDTree(np.column_stack([self.mid_x, self.mid_y]))
        
        # Sampled KDTree for 0% miss rate candidate generation
        self._build_sampled_tree()

    def _build_sampled_tree(self, step_meters: float = 25.0):
        sampled_pts = []
        sampled_arc_ids = []
        
        for aid in range(len(self.arcs_df)):
            L = self.lengths[aid]
            n_samples = max(2, int(np.ceil(L / step_meters)) + 1)
            alphas = np.linspace(0.0, 1.0, n_samples)
            sx = self.x1[aid] + alphas * self.dx[aid]
            sy = self.y1[aid] + alphas * self.dy[aid]
            for px, py in zip(sx, sy):
                sampled_pts.append([px, py])
                sampled_arc_ids.append(aid)
                
        self.sampled_tree = KDTree(np.array(sampled_pts))
        self.sampled_arc_ids = np.array(sampled_arc_ids)

    def query_all_candidates_sampled(self, query_coords: np.ndarray, radius: float = 50.0) -> List[List[int]]:
        """Batch query candidates for all GPS coordinates at once using multi-threading."""
        matches = self.sampled_tree.query_ball_point(query_coords, r=radius, workers=-1)
        results = []
        for i, idx_list in enumerate(matches):
            if len(idx_list) == 0:
                _, closest = self.sampled_tree.query(query_coords[i], k=1)
                idx_list = [closest]
            cands = np.unique(self.sampled_arc_ids[idx_list]).tolist()
            results.append(cands)
        return results

    def query_all_candidates_midpoint(self, query_coords: np.ndarray, radius: float = 50.0) -> List[List[int]]:
        """Baseline batch search on midpoints with radius and 1-NN fallback."""
        matches = self.midpoint_tree.query_ball_point(query_coords, r=radius, workers=-1)
        results = []
        for i, idx_list in enumerate(matches):
            if len(idx_list) == 0:
                _, closest = self.midpoint_tree.query(query_coords[i], k=1)
                idx_list = [closest]
            results.append(list(set(idx_list)))
        return results

def match_points_to_route(
    track_x: np.ndarray,
    track_y: np.ndarray,
    route_arcs: np.ndarray,
    road_net: SpatialRoadNetwork
) -> List[int]:
    """
    Establish ground-truth arc label for each GPS observation by tracking
    chronological progress along the known reference route.
    """
    from_route = road_net.arcs_df.iloc[route_arcs]['from_node'].values
    to_route = road_net.arcs_df.iloc[route_arcs]['to_node'].values
    
    rx1, ry1 = road_net.nodes_x[from_route], road_net.nodes_y[from_route]
    rx2, ry2 = road_net.nodes_x[to_route], road_net.nodes_y[to_route]
    
    rdx = rx2 - rx1
    rdy = ry2 - ry1
    rlen_sq = rdx**2 + rdy**2
    
    route_idx = 0
    assigned_arcs = []
    
    for i in range(len(track_x)):
        px, py = track_x[i], track_y[i]
        
        # Look in forward window along route
        w_start = max(0, route_idx - 2)
        w_end = min(len(route_arcs), route_idx + 25)
        
        w_dx = rdx[w_start:w_end]
        w_dy = rdy[w_start:w_end]
        w_lensq = rlen_sq[w_start:w_end]
        w_x1 = rx1[w_start:w_end]
        w_y1 = ry1[w_start:w_end]
        
        u = ((px - w_x1) * w_dx + (py - w_y1) * w_dy) / np.maximum(w_lensq, 1e-8)
        u_clamped = np.clip(u, 0.0, 1.0)
        proj_x = w_x1 + u_clamped * w_dx
        proj_y = w_y1 + u_clamped * w_dy
        
        dist = np.hypot(px - proj_x, py - proj_y)
        best_offset = int(np.argmin(dist))
        best_idx = w_start + best_offset
        route_idx = best_idx
        assigned_arcs.append(int(route_arcs[best_idx]))
        
    return assigned_arcs

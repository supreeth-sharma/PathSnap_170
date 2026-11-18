import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

class DirectedRoadGraph:
    """
    Constructs and manages a directed road network graph for topological route matching.
    Edges correspond to .arcs rows and vertices correspond to .nodes rows.
    """
    def __init__(self, nodes_df: pd.DataFrame, arcs_df: pd.DataFrame, lon0: float, lat0: float):
        self.arcs_df = arcs_df.reset_index(drop=True)
        self.nodes_df = nodes_df.reset_index(drop=True)
        self.lon0 = lon0
        self.lat0 = lat0
        
        # Metric coordinates
        lat0_rad = np.radians(lat0)
        lon0_rad = np.radians(lon0)
        R = 6371000.0
        
        self.nodes_x = R * (np.radians(self.nodes_df['lon'].values) - lon0_rad) * np.cos(lat0_rad)
        self.nodes_y = R * (np.radians(self.nodes_df['lat'].values) - lat0_rad)
        
        # Build NetworkX DiGraph
        self.graph = nx.DiGraph()
        
        # Add nodes
        for nid in range(len(self.nodes_df)):
            self.graph.add_node(nid, x=self.nodes_x[nid], y=self.nodes_y[nid])
            
        # Arcs endpoints and lengths
        self.arc_from = self.arcs_df['from_node'].values
        self.arc_to = self.arcs_df['to_node'].values
        
        dx = self.nodes_x[self.arc_to] - self.nodes_x[self.arc_from]
        dy = self.nodes_y[self.arc_to] - self.nodes_y[self.arc_from]
        self.arc_lengths = np.hypot(dx, dy)
        
        # Add edges
        for aid in range(len(self.arcs_df)):
            u = int(self.arc_from[aid])
            v = int(self.arc_to[aid])
            L = float(self.arc_lengths[aid])
            # Handle multi-edges or existing edge by keeping shortest length
            if self.graph.has_edge(u, v):
                if L < self.graph[u][v]['weight']:
                    self.graph[u][v]['weight'] = L
                    self.graph[u][v]['arc_id'] = aid
            else:
                self.graph.add_edge(u, v, weight=L, arc_id=aid)
                
        # Shortest path distance cache: (u, v) -> distance
        self._sp_cache: Dict[Tuple[int, int], float] = {}

    def get_node_distance(self, u: int, v: int, cutoff_meters: float = 2000.0) -> float:
        """
        Dijkstra shortest path distance between vertex u and vertex v.
        Returns float distance or inf if unreachable.
        """
        if u == v:
            return 0.0
        key = (u, v)
        if key in self._sp_cache:
            return self._sp_cache[key]
            
        try:
            d = nx.dijkstra_path_length(self.graph, u, v, weight='weight')
            if d > cutoff_meters:
                d = float('inf')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            d = float('inf')
            
        self._sp_cache[key] = d
        return d

    def get_arc_network_distance(
        self,
        arc_i: int,
        arc_j: int,
        proj_i: float = 0.5,
        proj_j: float = 0.5,
        cutoff_meters: float = 2000.0
    ) -> float:
        """
        Calculates network driving distance from vehicle position on arc_i (proj_i)
        to vehicle position on arc_j (proj_j).
        """
        L_i = float(self.arc_lengths[arc_i])
        L_j = float(self.arc_lengths[arc_j])
        
        if arc_i == arc_j:
            # Traveling along the same segment
            delta_proj = proj_j - proj_i
            if delta_proj >= -0.05: # allow small GPS jitter backwards on same arc
                return max(0.0, delta_proj) * L_i
            else:
                # Vehicle moved backwards against road direction
                return float('inf')
                
        u_i = int(self.arc_from[arc_i])
        v_i = int(self.arc_to[arc_i])
        u_j = int(self.arc_from[arc_j])
        v_j = int(self.arc_to[arc_j])
        
        rem_i = max(0.0, 1.0 - proj_i) * L_i
        init_j = max(0.0, proj_j) * L_j
        
        # Direct successor
        if v_i == u_j:
            return rem_i + init_j
            
        # Path query from v_i to u_j
        path_dist = self.get_node_distance(v_i, u_j, cutoff_meters=cutoff_meters)
        if np.isinf(path_dist):
            return float('inf')
            
        return rem_i + path_dist + init_j

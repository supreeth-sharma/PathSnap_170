import os
import folium
from folium import plugins
from typing import List, Dict, Optional, Tuple

def create_folium_map(
    gps_points: List[List[float]],
    predicted_coords: List[List[float]],
    ground_truth_coords: Optional[List[List[float]]] = None,
    road_network_segments: Optional[List[List[List[float]]]] = None,
    title: str = "PathSnap Route Reconstruction",
    save_html_path: Optional[str] = None
) -> folium.Map:
    """
    Renders an interactive Folium map with 4 distinct toggleable layers:
    1. Raw GPS Observations (Blue CircleMarkers)
    2. Road Network Geometry (Gray LineSegments)
    3. Ground Truth Route (Green PolyLine)
    4. Reconstructed Route (Red PolyLine)
    """
    if len(gps_points) == 0:
        center_lat, center_lon = 49.85, 8.60
    else:
        # Note: points in GeoJSON are [lon, lat], folium expects [lat, lon]
        center_lat = float(np_mean([p[1] for p in gps_points]))
        center_lon = float(np_mean([p[0] for p in gps_points]))
        
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="OpenStreetMap"
    )
    
    # Layer 1: Road Network (Gray)
    if road_network_segments and len(road_network_segments) > 0:
        fg_roads = folium.FeatureGroup(name="1. Road Network Geometry (Gray)", show=True)
        for seg in road_network_segments:
            # seg is [[lon1, lat1], [lon2, lat2]] -> folium wants [[lat1, lon1], [lat2, lon2]]
            coords = [[seg[0][1], seg[0][0]], [seg[1][1], seg[1][0]]]
            folium.PolyLine(coords, color="#9CA3AF", weight=1.5, opacity=0.6).add_to(fg_roads)
        fg_roads.add_to(m)
        
    # Layer 2: Raw GPS Points (Blue)
    fg_gps = folium.FeatureGroup(name="2. Raw GPS Trajectory (Blue)", show=True)
    # Subsample if too dense for fast rendering
    step = max(1, len(gps_points) // 800)
    for i in range(0, len(gps_points), step):
        pt = gps_points[i]
        folium.CircleMarker(
            location=[pt[1], pt[0]],
            radius=3.5,
            color="#2563EB",
            fill=True,
            fill_color="#3B82F6",
            fill_opacity=0.8,
            popup=f"GPS #{i} (Lon: {pt[0]:.5f}, Lat: {pt[1]:.5f})"
        ).add_to(fg_gps)
    fg_gps.add_to(m)
    
    # Layer 3: Ground Truth Route (Green)
    if ground_truth_coords and len(ground_truth_coords) > 0:
        fg_gt = folium.FeatureGroup(name="3. Ground Truth Reference (Green)", show=True)
        gt_folium = [[p[1], p[0]] for p in ground_truth_coords]
        folium.PolyLine(
            gt_folium,
            color="#10B981",
            weight=5.0,
            opacity=0.75,
            tooltip="Ground Truth Reference Route"
        ).add_to(fg_gt)
        fg_gt.add_to(m)
        
    # Layer 4: Reconstructed Route (Red)
    if predicted_coords and len(predicted_coords) > 0:
        fg_pred = folium.FeatureGroup(name="4. Reconstructed Route (Red)", show=True)
        pred_folium = [[p[1], p[0]] for p in predicted_coords]
        folium.PolyLine(
            pred_folium,
            color="#EF4444",
            weight=3.5,
            opacity=0.9,
            dash_array="6, 6",
            tooltip="PathSnap Reconstructed Route"
        ).add_to(fg_pred)
        fg_pred.add_to(m)
        
    # Layer Control
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Add floating custom legend
    legend_html = f'''
     <div style="position: fixed; 
     bottom: 30px; left: 30px; width: 250px; height: 160px; 
     border:2px solid #E5E7EB; z-index:9999; font-size:12px;
     background-color: white; opacity: 0.95; border-radius: 8px;
     padding: 10px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
     <b style="color:#1E3A8A; font-size:13px;">{title}</b><br>
     <i style="color:#6B7280; font-size:10px;">PathSnapppp ....</i><hr style="margin:4px 0;">
     <span style="color:#2563EB;">&#9679;</span> Raw GPS Observations (1 Hz)<br>
     <span style="color:#9CA3AF;">&#9644;&#9644;</span> Directed Road Network<br>
     <span style="color:#10B981;">&#9644;&#9644;</span> Ground Truth Reference<br>
     <span style="color:#EF4444;">&#9644; - &#9644;</span> PathSnap Reconstructed Route
     </div>
     '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    if save_html_path:
        os.makedirs(os.path.dirname(save_html_path), exist_ok=True)
        m.save(save_html_path)
        print(f"Saved interactive map to {save_html_path}")
        
    return m

def np_mean(vals):
    return sum(vals) / len(vals) if len(vals) > 0 else 0.0

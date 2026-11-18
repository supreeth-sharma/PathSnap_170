import os
import sys
import json
import requests
import streamlit as st
import streamlit.components.v1 as components

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.service.visualizer import create_folium_map

st.set_page_config(
    page_title="PathSnap | Distributed Map-Matching & Route Reconstruction",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 26px;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 2px;
    }
    .sub-header {
        font-size: 14px;
        color: #4B5563;
        margin-bottom: 18px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-val {
        font-size: 24px;
        font-weight: 700;
        color: #1E3A8A;
    }
    .metric-lbl {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #64748B;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">PathSnap: Distributed Map-Matching & Route Reconstruction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-Time Trajectory Map-Matching & Evaluation Dashboard</div>', unsafe_allow_html=True)
# Sidebar Configuration
st.sidebar.header("🌐 Distributed Connection")
st.sidebar.markdown("Configure Laptop 1 API host & port:")

api_host = st.sidebar.text_input("Laptop 1 IP / Hostname", value="127.0.0.1")
api_port = st.sidebar.number_input("Port", value=8000, min_value=1024, max_value=65535)
base_url = f"http://{api_host}:{api_port}"

# Health check
if st.sidebar.button("🔍 Check API Status"):
    try:
        r = requests.get(f"{base_url}/health", timeout=3)
        if r.status_code == 200:
            st.sidebar.success(f"Connected to Laptop 1! ({r.json().get('role', 'API Online')})")
        else:
            st.sidebar.error(f"API returned status {r.status_code}")
    except Exception as e:
        st.sidebar.error(f"Cannot reach Laptop 1: {str(e)}")

st.sidebar.markdown("---")
st.sidebar.header("🚘 Evaluation Trip Selection")

# 10 Reserved Student-Test Evaluation Trips
EVAL_TRIPS = [
    "00000000", "00000007", "00000009", "00000040", "00000052",
    "00000056", "00000065", "00000076", "00000077", "00000091"
]

selected_trip = st.sidebar.selectbox("Select Trip ID", EVAL_TRIPS, index=0)

run_btn = st.sidebar.button("🚀 Reconstruct Route", type="primary", use_container_width=True)

# Main Dashboard View
if run_btn or 'last_result' in st.session_state:
    if run_btn:
        with st.spinner(f"Sending reconstruction request for Trip {selected_trip} to Laptop 1 ({base_url})..."):
            try:
                payload = {"trip_id": selected_trip}
                res = requests.post(f"{base_url}/reconstruct_trip", json=payload, timeout=60)
                if res.status_code == 200:
                    st.session_state['last_result'] = res.json()
                    st.session_state['last_trip'] = selected_trip
                else:
                    st.error(f"Server Error {res.status_code}: {res.text}")
                    st.stop()
            except Exception as e:
                st.error(f"Failed to communicate with Laptop 1 API: {str(e)}")
                st.stop()
                
    data = st.session_state.get('last_result', {})
    metrics = data.get('metrics', {})
    geojson = data.get('geojson', {})
    
    st.subheader(f"Results for Trip: `{st.session_state.get('last_trip', selected_trip)}`")
    
    # 4 Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{metrics.get('pointwise_accuracy', 0.0) * 100:.1f}%</div>
            <div class="metric-lbl">Pointwise Top-1 Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{metrics.get('macro_f1', 0.0):.4f}</div>
            <div class="metric-lbl">Macro F1 Score</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{metrics.get('route_continuity', 0.0):.1f}%</div>
            <div class="metric-lbl">Route Continuity Rate</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{metrics.get('inference_time_ms', 0.0):.0f} ms</div>
            <div class="metric-lbl">End-to-End Latency</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Folium Interactive Map
    st.markdown("### 🗺️ Interactive Multi-Layer Trajectory Map")
    folium_map = create_folium_map(
        gps_points=geojson.get('gps_points', []),
        predicted_coords=geojson.get('predicted_route', []),
        ground_truth_coords=geojson.get('ground_truth_route', []),
        title=f"PathSnap Reconstruction — Trip {st.session_state.get('last_trip', selected_trip)}"
    )
    
    # Render map
    components.html(folium_map._repr_html_(), height=550, scrolling=False)
    
    # Collapsible details
    with st.expander("📋 View Predicted Arc Sequence & API Response Details"):
        st.write(f"Total Predicted Arcs: **{len(data.get('predicted_arcs', []))}**")
        st.write(f"First 20 Predicted Arcs: `{data.get('predicted_arcs', [])[:20]}`")
        st.json(metrics)
else:
    st.info("Select a trip form the side bar and click 'Reconstruct Route' meow")

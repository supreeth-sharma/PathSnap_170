# PathSnap: Distributed Map-Matching & Route Reconstruction

> **PES University | Department of Computer Science and Engineering | RR Campus**  
> **Course:** UE24CS352A Machine Learning Hackathon  
> **Challenge:** PathSnap Distributed Trajectory Map-Matching System  

---

## 📌 Executive Summary

Urban GPS tracking systems inherently suffer from multipath reflections, tall building occlusion, atmospheric disturbances, and sensor noise (nominally 1 Hz). Consequently, vehicles travelling along roadways frequently appear to drift onto parallel service roads, jump across city blocks, or deviate significantly from their true topological trajectories.

**PathSnap** is an end-to-end distributed machine learning and sequential trajectory decoding system that snaps noisy GPS coordinates onto a directed road network. It employs:
1. **Local Cartesian Metric Projection** ($R = 6,371,000$ m) to eliminate the 35.5% anisotropic spatial distortion inherent in unprojected WGS84 degree coordinates.
2. **Segment-Sampled KDTree Spatial Indexing** that cuts candidate miss rate from **32.6%** (naive midpoint query) to **0.00%**.
3. **Group-Aware Machine Learning Classification** across 5 distinct algorithm families evaluated using `GroupKFold(n_splits=5, groups=trip_id)` to strictly prevent temporal and spatial data leakage.
4. **Bayesian Hyperparameter Optimization** using Optuna (35 trials) maximizing validation Macro F1.
5. **Hidden Markov Model (HMM) Viterbi Trellis Decoding** (Newson & Krumm transition model with NetworkX Dijkstra shortest-path caching) achieving **>98% route continuity**.
6. **Two-Laptop Distributed Deployment Architecture** with a FastAPI model server on Laptop 1 and an interactive Streamlit UI on Laptop 2, enforcing **zero ground-truth label leakage**.

---

## 📂 Repository Structure

```
.
├── Student folder/
│   ├── train/                 # 80 training trips (tracks, nodes, arcs, routes)
│   └── student_test/          # 10 development-stage test trips
├── data/
│   └── processed/
│       ├── train_features.parquet      # Contract 1 training dataset (532k candidate rows)
│       ├── test_features.parquet       # Contract 1 student-test dataset (209k candidate rows)
│       ├── eda_summary.json            # EDA partition statistics and gap analysis
│       ├── pipeline_summary.json       # Pipeline class balance & run summary
│       └── candidate_miss_rate_comparison.json # Baseline vs Sampled miss rate benchmark
├── src/
│   ├── data_engine/           # Member 1: Spatial & Data Engine
│   │   ├── spatial.py         # Cartesian projection, angular differences, KDTree spatial indices
│   │   ├── feature_builder.py # Feature extraction complying with Contract 1
│   │   ├── pipeline.py        # Batch data processing pipeline
│   │   └── eda_stats.py       # Dataset audit & statistics generator
│   ├── models/                # Member 2: Modeling & Bayesian Optimization
│   │   ├── train_models.py    # 5 algorithm families with 5-fold GroupKFold evaluation
│   │   ├── bayesian_tune.py   # Optuna Bayesian hyperparameter optimization (35 trials)
│   │   └── inference.py       # Contract 2 inference wrapper (predict_candidate_probs)
│   ├── decoder/               # Member 3: Network Graph & HMM Viterbi Decoder
│   │   ├── graph_builder.py   # Directed NetworkX graph with Dijkstra caching
│   │   └── hmm_viterbi.py     # Newson & Krumm HMM Viterbi Trellis decoder (Contract 3)
│   └── service/               # Member 4: Two-Laptop Distributed API & UI
│       ├── api.py             # Laptop 1 FastAPI backend (Contract 4, server-side evaluation)
│       ├── client_ui.py       # Laptop 2 Streamlit evaluation interface
│       └── visualizer.py      # Folium 4-layer interactive map generator
├── artifacts/
│   ├── best_model.joblib               # Tuned LightGBM model artifact
│   ├── baseline_best_model.joblib      # Baseline best model
│   ├── optuna_study.joblib             # Full Optuna Bayesian trial study
│   ├── model_comparison_results.json   # 5-family cross-validation benchmark
│   └── tuning_results.json             # Tuned vs untuned comparison metrics
├── notebooks/
│   └── eda.ipynb              # Exploratory Data Analysis & Spatial Preprocessing notebook
├── reports/
│   ├── figures/               # ROC curves, PR curves, Confusion Matrices, Optuna history
│   ├── maps/                  # Exported interactive Folium HTML maps
│   └── PathSnap_Technical_Report.pdf   # Publication-quality final technical report
├── tests/
│   ├── test_contracts.py      # Comprehensive unit tests validating Contracts 1, 2, 3, 4
│   └── test_end_to_end.py     # End-to-end trajectory reconstruction test
├── requirements.txt           # Pinned python dependencies
└── README.md                  # Master project documentation
```

---

## 📐 Mathematical Formulation

### 1. Metric Cartesian Coordinate Projection
Because degrees of longitude converge poleward ($R \cdot \cos(\phi) \cdot \Delta \lambda$), using Euclidean distance in degrees introduces an anisotropic spatial distortion of **35.5%** at $49.85^\circ$N ($1^\circ$ lat $\approx 111,195$ m vs $1^\circ$ lon $\approx 71,700$ m). We project coordinates to local Cartesian meters centered at $(\text{lon}_0, \text{lat}_0)$:
$$x = R \cdot (\text{lon} - \text{lon}_0) \cdot \cos(\text{lat}_0) \cdot \frac{\pi}{180}$$
$$y = R \cdot (\text{lat} - \text{lat}_0) \cdot \frac{\pi}{180}$$
where $R = 6,371,000$ meters.

### 2. Candidate Road Arc Geometry
For each candidate road segment $a_i = (v_{\text{from}} \to v_{\text{to}})$ and GPS point $p_t = (x_t, y_t)$:
- **Scalar Projection Ratio ($u$):**
  $$u = \frac{(p_t - v_{\text{from}}) \cdot (v_{\text{to}} - v_{\text{from}})}{\|v_{\text{to}} - v_{\text{from}}\|^2}, \quad u_{\text{clamped}} = \text{clip}(u, 0.0, 1.0)$$
- **Perpendicular Distance ($d_\perp$):**
  $$p_{\text{proj}} = v_{\text{from}} + u_{\text{clamped}} \cdot (v_{\text{to}} - v_{\text{from}}), \quad d_\perp = \|p_t - p_{\text{proj}}\|$$
- **Heading Difference ($|\Delta \theta|$):**
  $$\theta_{\text{gps}} = \text{atan2}(\Delta x, \Delta y), \quad \theta_{\text{arc}} = \text{atan2}(x_{\text{to}} - x_{\text{from}}, y_{\text{to}} - y_{\text{from}})$$
  $$|\Delta \theta| = \min(|\theta_{\text{gps}} - \theta_{\text{arc}}|, 360^\circ - |\theta_{\text{gps}} - \theta_{\text{arc}}|) \in [0^\circ, 180^\circ]$$

### 3. HMM Viterbi Sequence Reconstruction (Newson & Krumm 2009)
- **Hidden States:** Candidate road segments $\{s_1, \dots, s_K\}$ at time step $t$.
- **Emission Probability:** Softmax temperature scaling over ML classifier confidence:
  $$P(z_t | a_i) = \frac{\exp(\text{Score}(z_t, a_i) / T)}{\sum_k \exp(\text{Score}(z_t, a_k) / T)} \quad (T = 0.5)$$
- **Transition Probability:** Exponential decay on network driving distance discrepancy:
  $$P(a_j | a_i) = \frac{1}{\beta} \exp\left( - \frac{|d_{\text{network}}(a_i, a_j) - d_{\text{great\_circle}}(z_{t-1}, z_t)|}{\beta} \right)$$
  If no directed path exists in the road graph, assign disconnected penalty $\epsilon = 10^{-7}$ ($\ln \epsilon = -16.1$).
- **Log-Viterbi Trellis Recurrence:**
  $$V_t(j) = \max_i [V_{t-1}(i) + \ln P(a_j | a_i)] + \ln P(z_t | a_j)$$

---

## 🤝 Inter-Member Data Contracts

The codebase strictly adheres to four modular data contracts:

### Contract 1: Candidate Parquet Schema (Member 1 $\to$ Members 2 & 3)
```
train_features.parquet Columns:
---------------------------------------------------------------------------
trip_id          : string   | e.g. '00000001'
pt_idx           : int32    | 0-based index of observation in .track
timestamp        : float64  | GPS timestamp in seconds
arc_id           : int32    | Candidate road segment ID from .arcs
dist_to_arc      : float32  | Perpendicular distance from GPS to arc (m)
heading_diff     : float32  | |GPS heading - Arc heading| in degrees [0, 180]
gps_speed        : float32  | Instantaneous speed in m/s
arc_length       : float32  | Length of candidate road segment (m)
projection_ratio : float32  | Scalar projection along segment [0, 1]
is_match         : int8     | Target label: 1 if arc_id == ground truth, else 0
```

### Contract 2: Model Inference Signature (Member 2 $\to$ Members 3 & 4)
```python
def predict_candidate_probs(features_df: pd.DataFrame) -> np.ndarray:
    """
    Input: DataFrame with columns ['dist_to_arc', 'heading_diff', 'gps_speed', 'arc_length', 'projection_ratio']
    Output: 1D NumPy array of predicted probabilities P(match=1) for each candidate.
    """
```

### Contract 3: Route Reconstruction Signature (Member 3 $\to$ Member 4)
```python
def reconstruct_route(trip_id: str, track_df, nodes_df, arcs_df, route_df=None) -> dict:
    """
    Runs KDTree candidate generation + ML scoring + Viterbi HMM decoding.
    Returns:
    {
        'trip_id': '00000000',
        'predicted_arcs': [2641, 64973, 133948, ...],
        'route_continuity_pct': 98.4,
        'pointwise_top1_acc': 0.956,
        'inference_time_ms': 310.0
    }
    """
```

### Contract 4: FastAPI REST JSON Payload (Member 4 Laptop 1 $\leftrightarrow$ Laptop 2)
```http
POST /reconstruct_trip
Request:  { "trip_id": "00000000" }
Response:
{
  "trip_id": "00000000",
  "metrics": {
    "pointwise_accuracy": 0.9562,
    "macro_f1": 0.9348,
    "route_continuity": 98.8,
    "inference_time_ms": 315.4
  },
  "predicted_arcs": [2641, 64973, 133948, ...],
  "geojson": {
    "gps_points": [[8.5952, 49.8440], [8.5953, 49.8444]],
    "predicted_route": [[8.5944, 49.8399], [8.5951, 49.8437]],
    "ground_truth_route": [[8.5944, 49.8399], [8.5951, 49.8437]]
  }
}
```

---

## 🚀 Setup & Execution Guide

### 1. Installation
Install all required libraries in your local Python environment:
```powershell
pip install -r requirements.txt
```

### 2. Feature Extraction Pipeline (Member 1)
Extract candidate features, compute miss rates, and generate parquet datasets:
```powershell
python -m src.data_engine.pipeline --max-train-trips 25 --max-test-trips 10
```

### 3. Model Training & Group-Aware Evaluation (Member 2)
Train all 5 algorithm families with 5-fold GroupKFold CV and generate comparison figures:
```powershell
python -m src.models.train_models
```

### 4. Bayesian Hyperparameter Optimization (Member 2)
Run 35 Bayesian optimization trials on LightGBM using Optuna:
```powershell
python -m src.models.bayesian_tune
```

### 5. Automated Contract & Integration Testing
Validate that all data contracts and end-to-end decoding pass:
```powershell
python -m unittest tests/test_contracts.py
python -m unittest tests/test_end_to_end.py
```

### 6. Distributed Two-Laptop Deployment (Member 4)

#### Laptop 1 (Model Server Host)
1. Find your local Wi-Fi IP address:
   ```powershell
   ipconfig
   # Look for IPv4 Address (e.g. 192.168.1.105)
   ```
2. Start the FastAPI backend server:
   ```powershell
   python -m src.service.api
   # Server starts at http://0.0.0.0:8000
   ```

#### Laptop 2 (Client UI)
1. Ensure Laptop 2 is connected to the same Wi-Fi / hotspot.
2. Launch the Streamlit evaluation dashboard:
   ```powershell
   streamlit run src/service/client_ui.py
   ```
3. Enter Laptop 1's IP address (e.g. `192.168.1.105`) and port `8000` in the sidebar.
4. Select any of the 10 evaluation trips from the dropdown and click **"Reconstruct Route"**.

---

## 📊 Benchmark Results

### 1. Spatial Candidate Miss Rate Benchmark
| Trip ID | GPS Points | Baseline Midpoint Miss % | Sampled KDTree Miss % | Avg Candidates / Pt |
|:-------:|:----------:|:------------------------:|:---------------------:|:-------------------:|
| 00000001 | 1,315 | 32.62% | **0.00%** | 12.1 |
| 00000003 | 821 | 19.61% | **0.00%** | 24.5 |
| 00000004 | 1,064 | 28.20% | **0.00%** | 7.6 |
| 00000005 | 1,045 | 35.12% | **0.00%** | 3.0 |

### 2. Multi-Model Comparison (5 Algorithm Families)
| Model | Algorithm Family | CV Macro F1 | Test Macro F1 | Observation Top-1 Acc | Log Loss | Train Time | Latency / 1k |
|:-----:|:----------------:|:-----------:|:-------------:|:---------------------:|:--------:|:----------:|:------------:|
| **Logistic Regression** | Linear / Probabilistic | 0.8124 ± 0.018 | 0.8042 | 0.8415 | 0.2351 | 3.8s | 0.85 ms |
| **Decision Tree** | Tree-based | 0.8872 ± 0.012 | 0.8791 | 0.9082 | 0.1620 | 2.1s | 0.42 ms |
| **Random Forest** | Ensemble | 0.9184 ± 0.009 | 0.9126 | 0.9345 | 0.1245 | 48.2s | 2.65 ms |
| **LightGBM** | Gradient Boosting | **0.9312 ± 0.007** | **0.9284** | **0.9478** | **0.0984** | 7.4s | 0.92 ms |
| **MLP Neural Net** | Neural Network | 0.8955 ± 0.014 | 0.8893 | 0.9142 | 0.1450 | 32.5s | 1.45 ms |

### 3. Sequential Route Reconstruction: Classifier-Only vs HMM Viterbi
| Evaluation Metric | Classifier-Only | Classifier + HMM Viterbi | Improvement |
|:------------------|:---------------:|:------------------------:|:-----------:|
| **Route Continuity %** | 72.4% | **98.8%** | **+26.4%** |
| **Pointwise Top-1 Acc** | 95.6% | **96.9%** | **+1.3%** |
| **Topological Inconsistencies** | 34 jumps | **0.4 jumps** | **-98.8%** |
| **Inference Latency** | 18 ms | 320 ms | Real-time capable |

---

## 🎯 Viva Defense & Interview Preparation

| Topic | Core Question | Definitive Technical Defense Answer |
|:------|:--------------|:------------------------------------|
| **Coordinate Math** | Why not use Euclidean distance on degrees? | Longitude lines converge toward the poles ($R \cos\phi \Delta\lambda$). At $49.85^\circ$N, $1^\circ$ lon is 71.7 km whereas $1^\circ$ lat is 111.2 km. Calculating Euclidean distance in degrees distorts space by 35.5%. We project to local Cartesian meters. |
| **Candidate Search** | Why did naive midpoint KDTree fail on long roads? | Road segments exceed 1,000m. Midpoint is $>50$m from road ends. Our segment-sampled KDTree samples road segments every 25m, reducing miss rate from 32.6% to 0.00%. |
| **Cross-Validation** | How did you prevent data leakage in CV? | Standard KFold splits randomly across timestamps, leaking contiguous GPS points into validation. We enforced GroupKFold strictly by `trip_id`, keeping entire trips in isolated folds. |
| **Evaluation Imbalance** | Why choose Macro F1 over simple Accuracy? | Severe class imbalance: ~9 to 15 negative candidates for every 1 true road arc. Simple accuracy is trivially inflated by predicting negatives. Macro F1 balances both classes equally. |
| **HMM Sequence Decoding** | How does HMM fix parallel street flips? | A pointwise classifier picks the lowest perpendicular distance; if GPS drifts 8m onto a parallel road, the classifier flips. HMM penalizes the network distance required to turn around, forcing continuity. |
| **Evaluation Integrity** | How is ground-truth leakage prevented during evaluation? | Ground truth `.route` files reside strictly on Laptop 1. The API computes accuracy server-side and sends back metrics. The client UI receives only predictions and summary scores. |

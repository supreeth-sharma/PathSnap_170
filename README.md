# PathSnap: Distributed Map-Matching & Route Reconstruction

**Team 170 | Machine Learning (UE24CS352A) Hackathon**  
*PES University — Department of Computer Science and Engineering*

---

## What is PathSnap?

When you drive through a city, tall buildings, tunnels, and trees cause GPS signals to bounce around and drift. A GPS ping might look like it's inside a building, in an oncoming lane, or on a parallel service road.

**PathSnap** takes these noisy GPS points and snaps them accurately onto the real road network. It uses a **fine-tuned LightGBM machine learning model** to score candidate road segments, combined with a **Hidden Markov Model (HMM) Viterbi decoder** to guarantee that the final reconstructed route is **100% physically connected and follows valid road traffic rules**.

The entire system is deployed across a **two-laptop distributed setup** over Wi-Fi, keeping ground-truth routes secure on the server with **zero data leakage**.

---

## Team Members & Roles

| Member Name | SRN | Core Role | What They Built |
| :--- | :--- | :--- | :--- |
| **Supreeth Sharma Y V** | `PES1UG24CS635` | **Data & Spatial Engine** | Metric Cartesian coordinate projection, 25m segment-sampled KDTree (0.00% miss rate), and Contract 1 feature builder. |
| **T Mohammed Saif** | `PES1UG24CS636` | **ML Models & Tuning** | Benchmarking 5 model families, 5-fold GroupKFold validation, 35-trial Optuna Bayesian optimization, and Logistic Regression deep-dive. |
| **Vishwas R Moodalappa** | `PES1UG24CS644` | **Graph Engine & HMM** | Directed NetworkX road graph, Newson & Krumm transition model, Dijkstra shortest path caching, and Viterbi trellis sequence decoder. |
| **Vismaya G** | `PES1UG24CS645` | **Distributed API & UI** | Laptop 1 FastAPI backend (port 8000, zero leakage), Laptop 2 Streamlit evaluator dashboard, and 4-layer interactive Folium maps. |

---

## How It Works (In 3 Simple Steps)

```
[ Raw Jittery GPS ] ──▶ [ 1. Spatial Search ] ──▶ [ 2. ML Scoring ] ──▶ [ 3. HMM Decoder ] ──▶ [ 100% Connected Route ]
```

1. **Spatial Candidate Search (Member 1):**  
   Converts GPS coordinates into flat metric meters so distance isn't distorted. Slices road segments every 25 meters so long highway roads are never missed (slashing candidate miss rates from **32.6% down to 0.00%**).
2. **Machine Learning Candidate Scoring (Member 2):**  
   Our fine-tuned **LightGBM** model evaluates nearby road candidates using distance, vehicle speed, and driving direction, outputting a calibrated confidence score for each road in just **0.45 ms**.
3. **Sequential Route Reconstruction (Member 3):**  
   Cars cannot teleport or drive the wrong way down one-way streets. Our **HMM Viterbi Trellis Decoder** uses the directed road graph to connect candidate roads into a smooth, legal path—boosting route continuity from **72.8% to a perfect 100.0%**.

---

## How to Run the Project

### 1. Installation
Clone the repository and install the dependencies:
```powershell
pip install -r requirements.txt
```

### 2. Running on Two Laptops (Distributed Setup)

#### Step A: On Laptop 1 (Model Server Host)
1. Find your Wi-Fi IPv4 address by running `ipconfig` (e.g., `10.20.201.76`).
2. Start the FastAPI backend server:
   ```powershell
   python -m src.service.api
   ```
   *(Server starts at `http://0.0.0.0:8000`)*

#### Step B: On Laptop 2 (Evaluator UI)
1. Make sure Laptop 2 is connected to the same Wi-Fi or hotspot.
2. Launch the Streamlit dashboard:
   ```powershell
   streamlit run src/service/client_ui.py
   ```
3. In the sidebar:
   * Enter Laptop 1's IP address (e.g., `10.20.201.76`) and port `8000`.
   * Click **Check API Status** (turns green when connected).
   * Pick any trip from the dropdown (e.g., `00000015` or `00000065`).
   * Click **Reconstruct Route**.

*(Note: If running both on a single laptop for testing, leave the IP as `127.0.0.1`)*

---

## 🔍 Useful Diagnostic Tools

* **Inspect Emission Probabilities for Any Trip:**
  ```powershell
  python src/decoder/list_emissions.py --trip 00000015 --points 3
  ```
* **Run Automated Contract & Integration Tests:**
  ```powershell
  python -m unittest tests/test_contracts.py
  python -m unittest tests/test_end_to_end.py
  ```

---

## Key Results

* **Candidate Miss Rate:** Reduced from **32.6% to 0.00%** using our 25m segment-sampled index.
* **Winning Model (LightGBM):** **80.54%** observation-level top-1 accuracy, **0.1011** log loss, and **0.45 ms** inference latency.
* **HMM Route Continuity:** **100.0%** continuous (0 broken jumps across all evaluation trips).
* **Overall PathSnap Accuracy:** **96.88%** after sequential Viterbi decoding.
* **Latency:** **~0.120 ms per GPS point** (~3100 ms per 2,500-point trip), beating the real-time requirement by over 40x.

---

## Repository Structure

```text
PathSnap_170/
├── evaluator_test/            # 10 evaluation trip folders (tracks, nodes, arcs, routes)
├── src/
│   ├── data_engine/           # Coordinate projection, KDTree & feature builder
│   ├── models/                # Model training, Optuna Bayesian tuning & inference
│   ├── decoder/               # NetworkX road graph, HMM Viterbi decoder & CLI tools
│   └── service/               # FastAPI backend server & Streamlit evaluator UI
├── artifacts/                 # Saved model weights (best_model.joblib) & study files
├── reports/                   # Performance charts, maps, and technical report
├── tests/                     # Automated unit and end-to-end test suites
├── report.pdf                 # Complete 12-page technical report
├── requirements.txt           # Python package dependencies
└── README.md                  # Project documentation
```

---

## Deliverables

* **Source Code:** Public GitHub repository: [https://github.com/supreeth-sharma/PathSnap_170]
* **Technical Report:** 12-page comprehensive report in `report.pdf'

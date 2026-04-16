# 🚦 City-Flow: FAANG-Level Traffic Prediction System

A production-ready **Spatio-Temporal Graph Neural Network (ST-GNN)** for real-time traffic speed prediction, trained on the **METR-LA** dataset (207 sensors, Los Angeles freeway network).

---

## 📊 Results

| Metric | Score |
|--------|-------|
| MAE    | ~4.2 mph |
| RMSE   | ~7.1 mph |
| MAPE   | ~11.3 %  |

---

## 🏗️ Architecture

```
Input (12 timesteps × 207 sensors)
        ↓
  Input Projection (Linear)
        ↓
  ST-Block 1 (Temporal Conv + Graph Conv)
        ↓
  ST-Block 2 (Temporal Conv + Graph Conv)
        ↓
  GRU (2 layers)
        ↓
  Output Projection
        ↓
Output (12 timesteps × 207 sensors)
```

---

## 📁 Project Structure

```
CITY-FLOW/
├── config/             # Hyperparameters and paths
├── data/               # Data loading, sequences, splits
├── models/             # ST-GNN architecture
├── training/           # Training loop, callbacks
├── inference/          # Prediction pipeline
├── api/                # FastAPI serving endpoint
├── utils/              # Metrics and visualization
├── main.py             # Full pipeline
└── run.py              # Inference only
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your data files
```
data/raw/METR-LA.h5
data/raw/adj_METR-LA.pkl
models/best_model.pth
models/scaler_final.pkl
```

### 3. Run inference
```bash
python run.py
```

### 4. Run full pipeline (train from scratch)
```bash
python main.py
```

### 5. Launch the dashboard
```bash
streamlit run dashboard.py
```

Then open: `http://localhost:8501`

### 6. Export to ONNX
```bash
python export_onnx.py
```

### 7. Start the API
```bash
python api/app.py
```

Then visit: `http://localhost:8000/docs`

---

## 🔌 API Usage

```python
import requests
import numpy as np

# Last 60 minutes of sensor readings (12 steps × 207 sensors)
speed_window = np.random.uniform(30, 70, size=(12, 207)).tolist()

response = requests.post(
    "http://localhost:8000/predict",
    json={"speed_window": speed_window}
)

print(response.json())
# {
#   "predictions": [[...207 speeds...], ...12 steps],
#   "message": "✅ Prediction successful — next 60 minutes forecasted"
# }
```

---

## 📦 Dataset

- **METR-LA**: Los Angeles highway traffic dataset
- 207 sensors, 5-minute intervals
- March 2012 – June 2012 (34,272 timesteps)
- Download: [DCRNN GitHub](https://github.com/liyaguang/DCRNN)

---

## 🛠️ Tech Stack

| Component | Tool |
|-----------|------|
| Model | PyTorch (ST-GNN) |
| API | FastAPI + Uvicorn |
| Experiment Tracking | MLflow |
| Model Optimization | ONNX |
| Visualization | Matplotlib |
| Config | YAML |

---

## 🗺️ Roadmap

- [x] ST-GNN model training
- [x] FastAPI inference endpoint
- [x] Streamlit live dashboard
- [x] MLflow experiment tracking
- [x] ONNX model optimization
- [ ] Model explainability (SHAP)
- [ ] Real-time Kafka streaming
- [ ] Docker deployment
- [ ] Cloud deployment (AWS SageMaker)

---

## 👤 Author

Built as a production ML portfolio project targeting FAANG-level engineering standards.

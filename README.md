# City-Flow

Traffic speed prediction using a Spatio‑Temporal Graph Neural Network (ST‑GNN) on the METR‑LA dataset (207 sensors).

## Results

| Metric | Value |
|--------|-------|
| MAE    | 4.2 mph |
| RMSE   | 7.1 mph |
| MAPE   | 11.3% |

## Quick Start

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Run inference (make sure data & model are in place)
python run.py

# 3. Start the API
python api/app.py
# Then open http://localhost:8000/docs

# 4. Launch dashboard
streamlit run dashboard.py

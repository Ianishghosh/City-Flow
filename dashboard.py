"""
dashboard.py — City-Flow Interactive Dashboard
Run with: streamlit run dashboard.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
import pickle
import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="City-Flow | Traffic Prediction",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
    }
    .stApp {
        background-color: #0a0e1a;
        color: #e0e6f0;
    }
    .block-container {
        padding-top: 1.5rem;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #111827, #1a2235);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 12px;
    }
    .metric-label {
        font-size: 11px;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #4a7fa5;
        margin-bottom: 6px;
        font-family: 'Space Mono', monospace;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 800;
        color: #ffffff;
        line-height: 1;
    }
    .metric-sub {
        font-size: 12px;
        color: #4a7fa5;
        margin-top: 4px;
        font-family: 'Space Mono', monospace;
    }

    /* Status badge */
    .badge-green  { background:#0d2b1e; color:#22c55e; border:1px solid #22c55e; border-radius:20px; padding:3px 12px; font-size:12px; font-family:'Space Mono',monospace; }
    .badge-yellow { background:#2b2100; color:#eab308; border:1px solid #eab308; border-radius:20px; padding:3px 12px; font-size:12px; font-family:'Space Mono',monospace; }
    .badge-red    { background:#2b0d0d; color:#ef4444; border:1px solid #ef4444; border-radius:20px; padding:3px 12px; font-size:12px; font-family:'Space Mono',monospace; }

    /* Section header */
    .section-title {
        font-size: 11px;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #4a7fa5;
        font-family: 'Space Mono', monospace;
        margin-bottom: 12px;
        border-bottom: 1px solid #1e3a5f;
        padding-bottom: 8px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0d1220;
        border-right: 1px solid #1e3a5f;
    }

    /* Streamlit overrides */
    .stSelectbox label, .stSlider label { color: #4a7fa5 !important; font-size: 12px; letter-spacing: 1px; }
    div[data-testid="stMetricValue"] { color: #ffffff !important; }
    .stPlotlyChart, .stpyplot { border-radius: 12px; overflow: hidden; }

    h1 { font-size: 28px !important; font-weight: 800 !important; color: #ffffff !important; }
    h2 { font-size: 18px !important; color: #a0b4cc !important; font-weight: 400 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

# Real METR-LA sensor approximate lat/lon (subset, others interpolated)
SENSOR_POSITIONS = {
    0:  (34.0522, -118.2437), 1:  (34.0480, -118.2600), 2:  (34.0550, -118.2300),
    3:  (34.0600, -118.2500), 4:  (34.0430, -118.2700), 5:  (34.0350, -118.2800),
    6:  (34.0650, -118.2200), 7:  (34.0700, -118.2100), 8:  (34.0400, -118.2900),
    9:  (34.0300, -118.3000), 10: (34.0750, -118.2000), 11: (34.0800, -118.1900),
    12: (34.0250, -118.3100), 13: (34.0900, -118.1800), 14: (34.0200, -118.3200),
}


def speed_to_color(speed, min_speed=0, max_speed=80):
    """Green = fast, Yellow = moderate, Red = slow."""
    ratio = np.clip((speed - min_speed) / (max_speed - min_speed), 0, 1)
    if ratio > 0.6:
        return "#22c55e", "FLOWING"
    elif ratio > 0.35:
        return "#eab308", "MODERATE"
    else:
        return "#ef4444", "CONGESTED"


def simulate_speed_data(n_sensors=207, n_steps=12, base_speed=55):
    """Generate realistic-looking speed data for demo mode."""
    np.random.seed(int(time.time()) % 100)
    data = np.random.normal(base_speed, 12, size=(n_steps, n_sensors))
    # Add congestion clusters
    congested = np.random.choice(n_sensors, size=30, replace=False)
    data[:, congested] *= np.random.uniform(0.3, 0.6, size=len(congested))
    return np.clip(data, 5, 80).astype(np.float32)


@st.cache_resource
def load_model_and_scaler(model_path, scaler_path):
    """Load model + scaler once and cache."""
    from config    import Config
    from inference import Predictor
    cfg       = Config()
    predictor = Predictor(cfg)
    return predictor


def get_predictions_demo(speed_window):
    """Demo predictions — add slight variation to input."""
    noise = np.random.normal(0, 2, size=speed_window.shape)
    future = speed_window + noise
    return np.clip(future, 5, 80)


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🚦 City-Flow")
    st.markdown("<p style='color:#4a7fa5;font-size:12px;font-family:Space Mono'>ST-GNN Traffic Intelligence</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("<div class='section-title'>Mode</div>", unsafe_allow_html=True)
    mode = st.radio("", ["🎭  Demo Mode", "🤖  Live Model"], label_visibility="collapsed")
    live_mode = mode == "🤖  Live Model"

    st.markdown("---")
    st.markdown("<div class='section-title'>Settings</div>", unsafe_allow_html=True)

    selected_sensor = st.selectbox(
        "SENSOR TO INSPECT",
        options=list(range(207)),
        format_func=lambda x: f"Sensor #{x:03d}"
    )

    base_speed = st.slider("BASE TRAFFIC SPEED (mph)", 20, 75, 55)
    auto_refresh = st.checkbox("AUTO-REFRESH (5s)", value=False)

    st.markdown("---")
    st.markdown("<div class='section-title'>Model Info</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:Space Mono;font-size:11px;color:#4a7fa5;line-height:2'>
    ARCHITECTURE &nbsp;ST-GNN<br>
    SENSORS &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;207<br>
    INPUT STEPS &nbsp;&nbsp;12 (60 min)<br>
    OUTPUT STEPS &nbsp;12 (60 min)<br>
    PARAMS &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;262,132<br>
    BEST VAL LOSS &nbsp;0.4019
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════

# Header
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown("# 🚦 City-Flow Traffic Intelligence")
    st.markdown("## Real-time prediction across 207 sensors · Los Angeles Freeway Network")
with col_status:
    st.markdown("<br>", unsafe_allow_html=True)
    now = pd.Timestamp.now().strftime("%H:%M:%S")
    st.markdown(f"""
    <div style='text-align:right'>
        <span class='badge-green'>● LIVE</span><br>
        <span style='font-family:Space Mono;font-size:11px;color:#4a7fa5'>{now}</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Generate / predict data ───────────────────────────────────────────
speed_window = simulate_speed_data(n_sensors=207, n_steps=12, base_speed=base_speed)

if live_mode:
    try:
        predictor = load_model_and_scaler("models/best_model.pth", "models/scaler_final.pkl")
        from config import Config
        cfg = Config()
        # cfg.adj_file is already an absolute path (resolved by Config._abs)
        with open(cfg.adj_file, "rb") as f:
            adj_raw = pickle.load(f, encoding='latin1')
            adj_mx  = adj_raw[2] if isinstance(adj_raw, (tuple, list)) and len(adj_raw) == 3 else adj_raw
        predictions = predictor.predict(speed_window, adj_mx)
        st.success("✅ Live model running!")
    except Exception as e:
        st.warning(f"⚠️ Model error: {e}. Using demo predictions.")
        st.cache_resource.clear()
        predictions = get_predictions_demo(speed_window)
else:
    predictions = get_predictions_demo(speed_window)

current_speeds = speed_window[-1, :]      # last known timestep
predicted_speeds = predictions[-1, :]    # predicted last step

# ── Top KPI row ───────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

congested_count  = int((current_speeds < 30).sum())
flowing_count    = int((current_speeds >= 55).sum())
avg_speed        = float(current_speeds.mean())
avg_pred_speed   = float(predicted_speeds.mean())
speed_delta      = avg_pred_speed - avg_speed

with k1:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Network Avg Speed</div>
        <div class='metric-value'>{avg_speed:.1f}</div>
        <div class='metric-sub'>mph · current</div>
    </div>""", unsafe_allow_html=True)

with k2:
    delta_color = "#22c55e" if speed_delta >= 0 else "#ef4444"
    delta_arrow = "▲" if speed_delta >= 0 else "▼"
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Predicted (60 min)</div>
        <div class='metric-value'>{avg_pred_speed:.1f}</div>
        <div class='metric-sub' style='color:{delta_color}'>{delta_arrow} {abs(speed_delta):.1f} mph change</div>
    </div>""", unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Congested Sensors</div>
        <div class='metric-value' style='color:#ef4444'>{congested_count}</div>
        <div class='metric-sub'>of 207 sensors</div>
    </div>""", unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Flowing Sensors</div>
        <div class='metric-value' style='color:#22c55e'>{flowing_count}</div>
        <div class='metric-sub'>speed ≥ 55 mph</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Main charts row ───────────────────────────────────────────────────
left_col, right_col = st.columns([1.4, 1])

with left_col:
    st.markdown("<div class='section-title'>Sensor Network — Current Speed Status</div>", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#0a0e1a")
    ax.set_facecolor("#0a0e1a")

    # Plot all 207 sensors as scatter
    # Use sensor index as proxy positions (arranged in grid)
    n = 207
    cols_grid = 23
    xs = np.array([i % cols_grid for i in range(n)], dtype=float)
    ys = np.array([i // cols_grid for i in range(n)], dtype=float)

    # Add slight jitter for natural look
    np.random.seed(42)
    xs += np.random.uniform(-0.3, 0.3, n)
    ys += np.random.uniform(-0.3, 0.3, n)

    colors = []
    sizes  = []
    for s in current_speeds:
        if s >= 55:
            colors.append("#22c55e")
            sizes.append(40)
        elif s >= 30:
            colors.append("#eab308")
            sizes.append(50)
        else:
            colors.append("#ef4444")
            sizes.append(70)

    # Draw connections (edges) between nearby sensors
    for i in range(n - 1):
        if abs(xs[i] - xs[i+1]) < 1.5 and abs(ys[i] - ys[i+1]) < 1.5:
            ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]],
                    color="#1e3a5f", linewidth=0.4, alpha=0.5, zorder=1)

    sc = ax.scatter(xs, ys, c=colors, s=sizes, zorder=2, alpha=0.9, linewidths=0)

    # Highlight selected sensor
    ax.scatter(xs[selected_sensor], ys[selected_sensor],
               c="white", s=150, zorder=3, marker="*")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#22c55e', markersize=8, label='Flowing (≥55 mph)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#eab308', markersize=8, label='Moderate (30-55 mph)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ef4444', markersize=8, label='Congested (<30 mph)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='white',   markersize=10, label=f'Selected: #{selected_sensor:03d}'),
    ]
    ax.legend(handles=legend_elements, loc="lower right",
              facecolor="#111827", edgecolor="#1e3a5f",
              labelcolor="#a0b4cc", fontsize=8)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("207 Sensors · LA Freeway Network", color="#4a7fa5",
                 fontsize=10, pad=10, fontfamily="monospace")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


with right_col:
    st.markdown("<div class='section-title'>Sensor Detail — Speed Forecast</div>", unsafe_allow_html=True)

    sensor_current = speed_window[:, selected_sensor]   # (12,)
    sensor_pred    = predictions[:, selected_sensor]     # (12,)
    s_color, s_status = speed_to_color(sensor_current[-1])

    # Status badge
    badge_class = "badge-green" if s_status == "FLOWING" else ("badge-yellow" if s_status == "MODERATE" else "badge-red")
    st.markdown(f"""
    <div style='margin-bottom:12px'>
        <span style='font-family:Space Mono;font-size:13px;color:white'>Sensor #{selected_sensor:03d}</span>
        &nbsp;&nbsp;<span class='{badge_class}'>{s_status}</span>
    </div>
    """, unsafe_allow_html=True)

    fig2, ax2 = plt.subplots(figsize=(5.5, 3.5), facecolor="#0a0e1a")
    ax2.set_facecolor("#111827")

    time_hist = [f"-{(12-i)*5}m" for i in range(12)]
    time_pred = [f"+{(i+1)*5}m" for i in range(12)]
    all_time  = time_hist + time_pred

    # Historical
    ax2.plot(range(12), sensor_current, color="#60a5fa", linewidth=2.5,
             label="Historical", marker="o", markersize=3)

    # Predicted
    ax2.plot(range(11, 24), [sensor_current[-1]] + list(sensor_pred),
             color="#f59e0b", linewidth=2.5, linestyle="--",
             label="Predicted", marker="o", markersize=3)

    # Divider
    ax2.axvline(x=11, color="#4a7fa5", linestyle=":", linewidth=1, alpha=0.7)
    ax2.text(11.2, ax2.get_ylim()[0] if ax2.get_ylim()[0] > 0 else 5,
             "NOW", color="#4a7fa5", fontsize=8, fontfamily="monospace")

    # Danger zone
    ax2.axhspan(0, 30, alpha=0.08, color="#ef4444")

    ax2.set_xticks(range(0, 24, 3))
    ax2.set_xticklabels([all_time[i] for i in range(0, 24, 3)],
                         fontsize=8, color="#4a7fa5", fontfamily="monospace")
    ax2.set_ylabel("Speed (mph)", color="#4a7fa5", fontsize=9)
    ax2.tick_params(colors="#4a7fa5")
    ax2.set_facecolor("#111827")

    for spine in ax2.spines.values():
        spine.set_color("#1e3a5f")

    ax2.legend(facecolor="#0a0e1a", edgecolor="#1e3a5f",
               labelcolor="#a0b4cc", fontsize=8, loc="upper left")
    ax2.grid(True, color="#1e3a5f", alpha=0.5, linewidth=0.5)

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # Mini stats for selected sensor
    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown(f"""
        <div class='metric-card' style='padding:12px'>
            <div class='metric-label'>Current</div>
            <div class='metric-value' style='font-size:22px;color:{s_color}'>{sensor_current[-1]:.1f}</div>
            <div class='metric-sub'>mph</div>
        </div>""", unsafe_allow_html=True)
    with mc2:
        pred_color, _ = speed_to_color(sensor_pred[-1])
        st.markdown(f"""
        <div class='metric-card' style='padding:12px'>
            <div class='metric-label'>In 60 min</div>
            <div class='metric-value' style='font-size:22px;color:{pred_color}'>{sensor_pred[-1]:.1f}</div>
            <div class='metric-sub'>mph predicted</div>
        </div>""", unsafe_allow_html=True)


# ── Bottom row: Speed distribution + Top congested ───────────────────
st.markdown("<br>", unsafe_allow_html=True)
b1, b2 = st.columns(2)

with b1:
    st.markdown("<div class='section-title'>Speed Distribution — All Sensors</div>", unsafe_allow_html=True)

    fig3, ax3 = plt.subplots(figsize=(6, 3), facecolor="#0a0e1a")
    ax3.set_facecolor("#111827")

    bins = np.linspace(0, 80, 25)
    ax3.hist(current_speeds,  bins=bins, alpha=0.7, color="#60a5fa", label="Current",   edgecolor="none")
    ax3.hist(predicted_speeds, bins=bins, alpha=0.5, color="#f59e0b", label="Predicted", edgecolor="none")

    ax3.axvline(30, color="#ef4444", linestyle="--", linewidth=1, alpha=0.7, label="Congestion threshold")
    ax3.set_xlabel("Speed (mph)", color="#4a7fa5", fontsize=9)
    ax3.set_ylabel("Sensor Count", color="#4a7fa5", fontsize=9)
    ax3.tick_params(colors="#4a7fa5")

    for spine in ax3.spines.values():
        spine.set_color("#1e3a5f")

    ax3.legend(facecolor="#0a0e1a", edgecolor="#1e3a5f", labelcolor="#a0b4cc", fontsize=8)
    ax3.grid(True, color="#1e3a5f", alpha=0.4, linewidth=0.5)

    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()


with b2:
    st.markdown("<div class='section-title'>Top 10 Most Congested Sensors</div>", unsafe_allow_html=True)

    worst_idx    = np.argsort(current_speeds)[:10]
    worst_speeds = current_speeds[worst_idx]
    worst_preds  = predicted_speeds[worst_idx]

    df_worst = pd.DataFrame({
        "Sensor": [f"#{i:03d}" for i in worst_idx],
        "Current (mph)": worst_speeds.round(1),
        "Predicted (mph)": worst_preds.round(1),
        "Status": [speed_to_color(s)[1] for s in worst_speeds]
    })

    def color_status(val):
        if val == "CONGESTED":
            return "color: #ef4444"
        elif val == "MODERATE":
            return "color: #eab308"
        return "color: #22c55e"

    st.dataframe(
        df_worst.style
            .applymap(color_status, subset=["Status"])
            .format({"Current (mph)": "{:.1f}", "Predicted (mph)": "{:.1f}"}),
        height=250,
        use_container_width=True
    )


# ── Auto refresh ──────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(5)
    st.rerun()

# ── Footer ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;font-family:Space Mono;font-size:10px;color:#2a4a6f;padding:10px'>
    CITY-FLOW · ST-GNN Traffic Intelligence · METR-LA Dataset · 207 Sensors · Los Angeles
</div>
""", unsafe_allow_html=True)
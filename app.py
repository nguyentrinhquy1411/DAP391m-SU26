import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import time
import math

from src.simulation import run_simulation
from src.simulator import EvidentialClassifierSimulator

# Page config
st.set_page_config(
    page_title="AES-RARR Ground Control Station",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme selector
theme_choice = st.sidebar.selectbox(
    "🎨 App Visual Theme",
    ["Dark Ocean", "Light Ocean"],
    index=0
)

# Custom CSS for Premium Look
import base64
import os

# Helper to load local image as base64
def get_base64_of_bin_file(bin_file):
    if os.path.exists(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return ""

if theme_choice == "Light Ocean":
    bg_base64 = get_base64_of_bin_file("assets/light_ocean_bg.png")
    app_bg = f"linear-gradient(180deg, rgba(220, 245, 247, 0.85) 0%, rgba(195, 235, 240, 0.92) 100%), url('data:image/png;base64,{bg_base64}')"
    sidebar_bg = f"linear-gradient(180deg, rgba(220, 245, 247, 0.95) 0%, rgba(185, 230, 238, 0.98) 100%), url('data:image/png;base64,{bg_base64}')"
    card_bg = "rgba(255, 255, 255, 0.7)"
    card_border = "1px solid rgba(0, 150, 255, 0.15)"
    warning_card_bg = "rgba(255, 230, 230, 0.8)"
    warning_card_border = "1px solid rgba(255, 75, 75, 0.25)"
    warning_card_bg_pulse_0 = "rgba(255, 220, 220, 0.7)"
    warning_card_bg_pulse_100 = "rgba(255, 200, 200, 0.85)"
    status_color = "#004d66"
    text_color_css = """
        :root {
            --primary-color: #007399 !important;
            --background-color: #f0fcfe !important;
            --secondary-background-color: #e0f7fa !important;
            --text-color: #0c3547 !important;
        }
        h1, h2, h3, h4, h5, h6, p, li, td, th, label, span, .status-text {
            color: #0c3547 !important;
        }
        div[data-testid="stWidgetLabel"] p, label {
            color: #0c3547 !important;
        }
        [data-testid="stMetricValue"] {
            color: #007399 !important;
        }
        [data-testid="stMetricLabel"] {
            color: #0b3c5d !important;
        }
        .metric-card, .warning-card {
            color: #0c3547 !important;
        }
        .metric-card * , .warning-card * {
            color: #0c3547 !important;
        }
        div[data-testid="stExpander"] {
            background-color: rgba(255, 255, 255, 0.6) !important;
            border: 1px solid rgba(0, 150, 255, 0.15) !important;
        }
        div[role="listbox"] {
            background-color: #ffffff !important;
            color: #0c3547 !important;
        }
        div[role="listbox"] * {
            background-color: #ffffff !important;
            color: #0c3547 !important;
        }
        div[data-baseweb="select"] {
            background-color: rgba(255, 255, 255, 0.8) !important;
            color: #0c3547 !important;
        }
        div[data-baseweb="select"] * {
            color: #0c3547 !important;
            background-color: transparent !important;
        }
        div[data-baseweb="popover"] {
            background-color: #ffffff !important;
            color: #0c3547 !important;
        }
        div[data-baseweb="popover"] * {
            background-color: transparent !important;
            color: #0c3547 !important;
        }
        input {
            background-color: rgba(255, 255, 255, 0.9) !important;
            color: #0c3547 !important;
        }
        div[data-testid="stNumberInput"] div {
            background-color: rgba(255, 255, 255, 0.9) !important;
            color: #0c3547 !important;
        }
        div[data-testid="stNumberInput"] input {
            color: #0c3547 !important;
        }
        div[data-testid="stNumberInput"] button {
            color: #0c3547 !important;
            background-color: rgba(0, 77, 102, 0.1) !important;
        }
        div[data-role="stSlider"] {
            color: #007399 !important;
        }
        div[data-testid="stSlider"] * {
            color: #0c3547 !important;
        }
        button[data-baseweb="tab"] {
            color: #0c3547 !important;
            background-color: transparent !important;
        }
        button[data-baseweb="tab"]:hover {
            color: #007399 !important;
        }
        button[aria-selected="true"] {
            color: #007399 !important;
            border-bottom-color: #007399 !important;
        }
        div[data-testid="stDataFrame"] * {
            color: #0c3547 !important;
            background-color: rgba(255, 255, 255, 0.5) !important;
        }
        table {
            background-color: rgba(255, 255, 255, 0.6) !important;
        }
        button[kind="secondary"] {
            background-color: rgba(255, 255, 255, 0.8) !important;
            color: #0c3547 !important;
            border: 1px solid rgba(0, 150, 255, 0.25) !important;
        }
        button[kind="secondary"]:hover {
            background-color: #007399 !important;
            color: #ffffff !important;
        }
    """
    grid_color = "rgba(0, 77, 102, 0.1)"
    text_color_plotly = "#0c3547"
    uav_color = "#004d66"
    shadow_color = "rgba(0, 77, 102, 0.15)"
    border_right_sidebar = "1px solid rgba(0, 77, 102, 0.15)"
else:
    bg_base64 = get_base64_of_bin_file("assets/dark_ocean_bg.png")
    app_bg = f"linear-gradient(180deg, rgba(10, 25, 47, 0.82) 0%, rgba(10, 17, 30, 0.92) 100%), url('data:image/png;base64,{bg_base64}')"
    sidebar_bg = f"linear-gradient(180deg, rgba(10, 25, 47, 0.93) 0%, rgba(5, 10, 18, 0.97) 100%), url('data:image/png;base64,{bg_base64}')"
    card_bg = "rgba(18, 32, 54, 0.65)"
    card_border = "1px solid rgba(255, 255, 255, 0.05)"
    warning_card_bg = "rgba(60, 20, 20, 0.65)"
    warning_card_border = "1px solid rgba(255, 255, 255, 0.05)"
    warning_card_bg_pulse_0 = "rgba(60, 20, 20, 0.55)"
    warning_card_bg_pulse_100 = "rgba(75, 22, 22, 0.75)"
    status_color = "#ffffff"
    text_color_css = """
        :root {
            --primary-color: #00d2ff !important;
            --background-color: #0a192f !important;
            --secondary-background-color: #0b1a30 !important;
            --text-color: #ffffff !important;
        }
        h1, h2, h3, h4, h5, h6, p, li, td, th, label, span, .status-text {
            color: #ffffff !important;
        }
        div[data-testid="stWidgetLabel"] p, label {
            color: #ffffff !important;
        }
        [data-testid="stMetricValue"] {
            color: #00d2ff !important;
        }
        [data-testid="stMetricLabel"] {
            color: #a0aec0 !important;
        }
        .metric-card, .warning-card {
            color: #ffffff !important;
        }
        .metric-card * , .warning-card * {
            color: #ffffff !important;
        }
        div[data-baseweb="select"] {
            background-color: rgba(18, 32, 54, 0.9) !important;
            color: #ffffff !important;
        }
        div[data-baseweb="select"] * {
            color: #ffffff !important;
            background-color: transparent !important;
        }
        div[role="listbox"] {
            background-color: #0b1a30 !important;
            color: #ffffff !important;
        }
        div[role="listbox"] * {
            background-color: #0b1a30 !important;
            color: #ffffff !important;
        }
        div[data-baseweb="popover"] {
            background-color: #0b1a30 !important;
            color: #ffffff !important;
        }
        div[data-baseweb="popover"] * {
            background-color: transparent !important;
            color: #ffffff !important;
        }
        input {
            background-color: rgba(18, 32, 54, 0.9) !important;
            color: #ffffff !important;
        }
        div[data-testid="stNumberInput"] div {
            background-color: rgba(18, 32, 54, 0.9) !important;
            color: #ffffff !important;
        }
        div[data-testid="stNumberInput"] input {
            color: #ffffff !important;
        }
        div[data-testid="stNumberInput"] button {
            color: #ffffff !important;
            background-color: rgba(255, 255, 255, 0.05) !important;
        }
        div[data-role="stSlider"] {
            color: #00d2ff !important;
        }
        div[data-testid="stSlider"] * {
            color: #ffffff !important;
        }
        div[data-testid="stExpander"] {
            background-color: rgba(18, 32, 54, 0.5) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
        }
        button[data-baseweb="tab"] {
            color: #ffffff !important;
            background-color: transparent !important;
        }
        button[data-baseweb="tab"]:hover {
            color: #00d2ff !important;
        }
        button[aria-selected="true"] {
            color: #00d2ff !important;
            border-bottom-color: #00d2ff !important;
        }
        div[data-testid="stDataFrame"] * {
            color: #ffffff !important;
            background-color: rgba(18, 32, 54, 0.4) !important;
        }
        table {
            background-color: rgba(18, 32, 54, 0.5) !important;
        }
        button[kind="secondary"] {
            background-color: rgba(18, 32, 54, 0.8) !important;
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
        }
        button[kind="secondary"]:hover {
            background-color: #00d2ff !important;
            color: #0a192f !important;
        }
    """
    grid_color = "rgba(255,255,255,0.08)"
    text_color_plotly = "#ffffff"
    uav_color = "#00ffff"
    shadow_color = "rgba(0, 0, 0, 0.3)"
    border_right_sidebar = "1px solid rgba(255, 255, 255, 0.05)"

# Custom CSS for Premium Look & Ocean Background
st.markdown(f"""
<style>
    /* Ocean Background Integration */
    [data-testid="stAppViewContainer"] {{
        background-image: {app_bg};
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    [data-testid="stHeader"] {{
        background: rgba(0, 0, 0, 0);
    }}
    [data-testid="stSidebar"] {{
        background-image: {sidebar_bg};
        background-size: cover;
        background-position: center;
        border-right: {border_right_sidebar};
    }}
    
    /* Glassmorphic Cards & UI Elements */
    .metric-card {{
        background-color: {card_bg};
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 18px;
        border-radius: 12px;
        border-left: 5px solid #00d2ff;
        border-top: {card_border};
        border-right: {card_border};
        border-bottom: {card_border};
        margin-bottom: 12px;
        box-shadow: 0 8px 32px 0 {shadow_color};
    }}
    .warning-card {{
        background-color: {warning_card_bg};
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 18px;
        border-radius: 12px;
        border-left: 5px solid #ff4b4b;
        border-top: {warning_card_border};
        border-right: {warning_card_border};
        border-bottom: {warning_card_border};
        margin-bottom: 12px;
        box-shadow: 0 8px 32px 0 {shadow_color};
        animation: pulse 3s infinite alternate;
    }}
    @keyframes pulse {{
        0% {{ background-color: {warning_card_bg_pulse_0}; box-shadow: 0 8px 32px 0 {shadow_color}; }}
        100% {{ background-color: {warning_card_bg_pulse_100}; box-shadow: 0 8px 32px 0 rgba(255, 75, 75, 0.1); }}
    }}
    .status-text {{
        font-weight: bold;
        font-size: 1.15rem;
        color: {status_color};
        letter-spacing: 0.5px;
    }}
    {text_color_css}
</style>
""", unsafe_allow_html=True)

# App Title & Description
st.markdown("# 🛸 AES-RARR Ground Control Station (GCS)")
st.markdown("""
*Active Evidential Sensing (AES) & Risk-Averse Rescue Routing (RARR) Framework for UAV Maritime Search and Rescue.*
""")

# Sidebar Controls
st.sidebar.markdown("### ⚙️ Mission Configuration")

mode = st.sidebar.selectbox(
    "UAV Routing & Decision Mode",
    ["aes_rarr", "no_branch", "static_R", "deterministic", "distance_router"],
    help="Select the framework configuration. 'aes_rarr' enables both evidential descent branches and uncertainty-adaptive tracking."
)

st.sidebar.markdown("#### 🧠 Decision Engine Parameters")
tau_unc = st.sidebar.slider(
    "Uncertainty Threshold (τ_unc)", 
    0.1, 1.0, 0.55, 0.05, 
    help="Triggers descent active sensing if epistemic uncertainty rises above this limit."
)
lambda_ra = st.sidebar.slider(
    "Risk-Aversion Factor (λ)", 
    0.0, 5.0, 0.8, 0.1, 
    help="Sets UCB confidence boost. High values prioritize exploration of uncertain targets; 0.0 ignores uncertainty in routing."
)
gamma_r = st.sidebar.slider(
    "Kalman Noise Scaling (γ_R)", 
    0.0, 10.0, 2.0, 0.5, 
    help="Scale factor for measurement covariance update. Scales KF observation noise when target is uncertain."
)

st.sidebar.markdown("#### ✈️ UAV Telemetry & Environment")
uav_speed = st.sidebar.slider("UAV Speed (m/s)", 5.0, 25.0, 10.0, 1.0)
descent_latency = st.sidebar.slider("Descent Latency (s)", 0.0, 5.0, 1.0, 0.5, help="Time cost incurred when descending from patrol altitude to verification altitude.")
sim_steps = st.sidebar.slider("Max Sim Steps", 10, 50, 30)

st.sidebar.markdown("#### 🌊 Target Setup")
num_victims = st.sidebar.slider("Number of Victims", 1, 10, 3)
occlusion_prob = st.sidebar.slider("Initial Occlusion Probability", 0.0, 1.0, 0.3, 0.1, help="Probability that targets start as 'occluded' by waves/foam.")
drift_x = st.sidebar.slider("Ocean Drift X (m/s)", -1.0, 1.0, 0.1, 0.05)
drift_y = st.sidebar.slider("Ocean Drift Y (m/s)", -1.0, 1.0, -0.05, 0.05)
ocean_drift = np.array([drift_x, drift_y])

seed = st.sidebar.number_input("Random Seed", value=42, step=1)

# Generate Victims consistently
@st.cache_data
def generate_victims(num_vics, seed_val, occ_prob):
    np.random.seed(seed_val)
    victim_init = {}
    
    # Victim 1 is always Drowning to ensure high-priority event
    pos1 = np.random.uniform(-60.0, 60.0, 2)
    while np.linalg.norm(pos1) < 15.0:
        pos1 = np.random.uniform(-60.0, 60.0, 2)
    victim_init[1] = {
        "pos": pos1,
        "class": 0, # Drowning
        "occluded": (np.random.rand() < occ_prob),
        "name": "Victim 1 (Drowning)"
    }
    
    classes = [0, 1, 2, 3]
    class_names = ["Drowning", "Floating", "Swimming", "PFD Floater"]
    
    for idx in range(2, num_vics + 1):
        pos = np.random.uniform(-70.0, 70.0, 2)
        while np.linalg.norm(pos) < 15.0:
            pos = np.random.uniform(-70.0, 70.0, 2)
        cls = np.random.choice(classes)
        victim_init[idx] = {
            "pos": pos,
            "class": cls,
            "occluded": (np.random.rand() < occ_prob),
            "name": f"Victim {idx} ({class_names[cls]})"
        }
    return victim_init

victim_init = generate_victims(num_victims, seed, occlusion_prob)

# Run simulations (Active configuration and Baseline Greedy Router)
@st.cache_data
def get_simulation_run(mode_val, u_speed, drift, t_unc, l_ra, g_r, steps, latent, vics):
    # Run the active mode with history
    results, branches, history = run_simulation(
        mode=mode_val,
        uav_speed=u_speed,
        ocean_drift=drift,
        tau_unc=t_unc,
        lambda_ra=l_ra,
        gamma_r=g_r,
        sim_steps=steps,
        descent_latency=latent,
        victim_init=vics,
        return_history=True
    )
    
    # Run baseline distance router (Greedy) under the same random seed conditions
    # For fair comparison, we need to reset the random state before baseline run
    np.random.seed(seed)
    base_results, base_branches = run_simulation(
        mode="distance_router",
        uav_speed=u_speed,
        ocean_drift=drift,
        tau_unc=t_unc,
        lambda_ra=l_ra,
        gamma_r=g_r,
        sim_steps=steps,
        descent_latency=latent,
        victim_init=vics,
        return_history=False
    )
    
    return results, branches, history, base_results, base_branches

results, branches, history, base_results, base_branches = get_simulation_run(
    mode, uav_speed, ocean_drift, tau_unc, lambda_ra, gamma_r, sim_steps, descent_latency, victim_init
)

# Playback controls
st.markdown("### ⏺️ Simulation Playback")
play_col1, play_col2, play_col3 = st.columns([1, 2, 5])

with play_col1:
    autoplay = st.checkbox("Auto Play 🔄", value=False)
with play_col2:
    speed_factor = st.slider("Animation Delay (s)", 0.05, 1.0, 0.3, 0.05)

max_steps = len(history)
if "step_counter" not in st.session_state:
    st.session_state.step_counter = 0

if autoplay:
    step_idx = st.session_state.step_counter
else:
    step_idx = st.slider(
        "Step Scrub", 
        0, 
        max_steps - 1, 
        value=min(st.session_state.step_counter, max_steps - 1), 
        format="Step %d"
    )
    st.session_state.step_counter = step_idx

current_state = history[step_idx]

# Layout Columns
col_left, col_right = st.columns([3, 2])

# Left column: Interactive Map
with col_left:
    st.markdown("#### 🗺️ Search Grid & UAV Tracking")
    
    # Build Map Figure
    fig = go.Figure()
    
    # Search grid boundaries
    fig.add_shape(type="rect", x0=-80, y0=-80, x1=80, y1=80,
                  line=dict(color="RoyalBlue", width=2, dash="dash"))
    
    # Extract historical UAV coordinates up to current step
    uav_path_x = [step_data["uav"][0] for step_data in history[:step_idx+1]]
    uav_path_y = [step_data["uav"][1] for step_data in history[:step_idx+1]]
    
    # UAV Trajectory Line
    fig.add_trace(go.Scatter(
        x=uav_path_x, y=uav_path_y,
        mode="lines+markers",
        name="UAV Path",
        line=dict(color="#00c0f2", width=3),
        marker=dict(size=4, color="#ffffff"),
    ))
    
    # Victims true and estimated positions
    vic_colors = {0: "#ff4b4b", 1: "#ffa500", 2: "#ffff00", 3: "#00ff00"}
    classes_labels = {0: "Drowning", 1: "Floating", 2: "Swimming", 3: "PFD Floater"}
    
    for vid, data in current_state["victims"].items():
        color = vic_colors[data["class"]]
        name = data["name"]
        
        # True position (represented by square marker)
        fig.add_trace(go.Scatter(
            x=[data["true_pos"][0]], y=[data["true_pos"][1]],
            mode="markers",
            name=f"{name} (True)",
            marker=dict(symbol="square-open" if data["occluded"] else "square", size=10, color=color),
            hoverinfo="text",
            hovertext=f"True Location: ({data['true_pos'][0]:.1f}, {data['true_pos'][1]:.1f})"
        ))
        
        # Estimated position (represented by uncertainty circle & centroid)
        u_val = data["u"]
        est_x, est_y = data["est_pos"][0], data["est_pos"][1]
        
        # Plotly circle representing uncertainty covariance radius
        cov_radius = (0.5 + 0.1 * u_val) * 8.0 # Scaled for visibility
        fig.add_shape(
            type="circle",
            xref="x", yref="y",
            x0=est_x - cov_radius, y0=est_y - cov_radius,
            x1=est_x + cov_radius, y1=est_y + cov_radius,
            line=dict(color=color, width=1, dash="dot"),
            fillcolor=color,
            opacity=0.15
        )
        
        # Estimated centroid point
        fig.add_trace(go.Scatter(
            x=[est_x], y=[est_y],
            mode="markers+text",
            text=[f"{data['name'].split(' ')[1]} (Est)"],
            textposition="top center",
            marker=dict(symbol="cross", size=8, color=color),
            name=f"{name} (Est)"
        ))
        
    # Sensor range circle around UAV (descent triggering radius <= 20m)
    uav_x, uav_y = current_state["uav"][0], current_state["uav"][1]
    fig.add_shape(
        type="circle",
        xref="x", yref="y",
        x0=uav_x - 20, y0=uav_y - 20,
        x1=uav_x + 20, y1=uav_y + 20,
        line=dict(color="#ffffff", width=1, dash="dash"),
        opacity=0.2
    )
    
    # Draw laser sensor beam from UAV to the current best target
    best_target = current_state["best_target"]
    if best_target is not None and best_target in current_state["victims"]:
        tgt_data = current_state["victims"][best_target]
        fig.add_trace(go.Scatter(
            x=[uav_x, tgt_data["est_pos"][0]],
            y=[uav_y, tgt_data["est_pos"][1]],
            mode="lines",
            line=dict(color="#ff33cc", width=1.5, dash="dash"),
            name="Sensor Scan Line",
            showlegend=False
        ))

    # UAV Current Position Marker
    fig.add_trace(go.Scatter(
        x=[uav_x], y=[uav_y],
        mode="markers",
        name="UAV Drone",
        marker=dict(symbol="triangle-up", size=18, color=uav_color, line=dict(color="#ffffff" if theme_choice == "Dark Ocean" else "#0c3547", width=2)),
        hoverinfo="text",
        hovertext=f"UAV: ({uav_x:.1f}, {uav_y:.1f})"
    ))
    
    fig.update_layout(
        xaxis=dict(range=[-85, 85], gridcolor=grid_color, zeroline=False, tickfont=dict(color=text_color_plotly)),
        yaxis=dict(range=[-85, 85], gridcolor=grid_color, zeroline=False, tickfont=dict(color=text_color_plotly)),
        width=700, height=580,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        font=dict(color=text_color_plotly)
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Right column: Telemetry & Dynamic Priorities
with col_right:
    st.markdown("#### 📊 Telemetry & Decision Metrics")
    
    # Active branch alert badge
    is_descended = (current_state["uav_altitude"] < 30.0)
    if is_descended:
        st.markdown(
            f"""<div class='warning-card'>
                <span class='status-text'>🚨 ACTIVE Evidential Sensing Triggered</span><br/>
                UAV has descended to <b>{current_state['uav_altitude']:.0f}m</b> to resolve occlusion and verify posture details.
            </div>""", 
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""<div class='metric-card'>
                <span class='status-text'>✈️ UAV Cruise Patrol Mode</span><br/>
                UAV Altitude: <b>{current_state['uav_altitude']:.0f}m</b> | Standard search altitude.
            </div>""", 
            unsafe_allow_html=True
        )
        
    # Numerical telemetries
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric("Flight Step", f"{step_idx + 1} / {max_steps}")
    with metric_col2:
        st.metric("TTR Time (s)", f"{current_state['cumulative_time']:.1f}")
    with metric_col3:
        st.metric("Branch descents", f"{current_state['branches']}")
        
    st.markdown("#### 🎯 Risk-UCB Priority Ranking")
    # Priority Queue table
    priority_rows = []
    for vid, data in current_state["victims"].items():
        if data["rescued"]:
            status_str = "✅ Rescued"
        else:
            status_str = f"🔍 Scan (Score: {data['score']:.3f})"
        
        priority_rows.append({
            "Victim": data["name"],
            "Posture": classes_labels[data["class"]],
            "Uncertainty (u)": round(data["u"], 3),
            "Expected Risk (er)": round(data["er"], 3),
            "Priority Score": round(data["score"], 4),
            "Status": status_str
        })
        
    p_df = pd.DataFrame(priority_rows).sort_values(by="Priority Score", ascending=False)
    st.dataframe(
        p_df, 
        column_config={
            "Priority Score": st.column_config.ProgressColumn("Priority Score", min_value=0.0, max_value=2.0, format="%.3f"),
            "Uncertainty (u)": st.column_config.ProgressColumn("Uncertainty (u)", min_value=0.0, max_value=1.0, format="%.2f")
        },
        use_container_width=True,
        hide_index=True
    )

# Lower Section Tabs
st.markdown("---")
tab1,tab2,tab3 = st.tabs(["📊 Live Evidential Subjective Logic (EDL)", "🏁 Comparative Performance Analysis", "🖼️ Evidential Visual Playground (Upload Image)"])

with tab1:
    st.markdown("### 🧬 Subjective Logic Evidence Vectors")
    st.markdown("Showing the Belief ($b$), Disbelief ($d$), and Epistemic Uncertainty ($u$) for each victim at the current step.")
    
    edl_cols = st.columns(num_victims)
    
    for idx, (vid, data) in enumerate(current_state["victims"].items()):
        with edl_cols[idx]:
            st.markdown(f"**{data['name']}**")
            st.markdown(f"Status: `{'Rescued' if data['rescued'] else ('Occluded' if data['occluded'] else 'Visible')}`")
            
            if "crop_img" in data and data["crop_img"] is not None:
                st.image(data["crop_img"], caption="UAV Camera Crop Input", use_container_width=True)
            
            # Draw Horizontal Bars
            # beliefs order in classifier: [Drowning, Floating, Swimming, PFD]
            # subjective logic: belief vector + disbelief + uncertainty = 1.0
            u_val = data["u"]
            beliefs_sum = sum(data["beliefs"])
            
            # Create a simple horizontal bar chart
            fig_bar = go.Figure()
            
            fig_bar.add_trace(go.Bar(
                y=["Evidence"], x=[data["beliefs"][0]], name="Belief (Drowning)",
                orientation='h', marker=dict(color='#ff4b4b')
            ))
            fig_bar.add_trace(go.Bar(
                y=["Evidence"], x=[data["beliefs"][1]], name="Belief (Floating)",
                orientation='h', marker=dict(color='#ffa500')
            ))
            fig_bar.add_trace(go.Bar(
                y=["Evidence"], x=[data["beliefs"][2] + data["beliefs"][3]], name="Belief (Other)",
                orientation='h', marker=dict(color='#00ff00')
            ))
            fig_bar.add_trace(go.Bar(
                y=["Evidence"], x=[u_val], name="Uncertainty (u)",
                orientation='h', marker=dict(color='#888888')
            ))
            
            fig_bar.update_layout(
                barmode='stack',
                height=150,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(range=[0, 1], showticklabels=True, tickfont=dict(color=text_color_plotly)),
                yaxis=dict(showticklabels=False),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=text_color_plotly)),
                font=dict(color=text_color_plotly)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    st.markdown("### 🏁 Live Trial Comparison vs. Distance Router (Greedy)")
    st.markdown("We run the exact same victim starting conditions through the baseline distance router (nearest-first) to demonstrate how the AES-RARR framework improves victim survival rate.")
    
    # Active run metrics
    worst_vsr_active = min([r["VSR"] for r in results])
    mean_vsr_active = np.mean([r["VSR"] for r in results])
    mean_ttr_active = np.mean([float(r["TTR (steps)"]) if not str(r["TTR (steps)"]).startswith(">") else float(r["TTR (steps)"][1:]) for r in results])
    
    # Baseline metrics
    worst_vsr_base = min([r["VSR"] for r in base_results])
    mean_vsr_base = np.mean([r["VSR"] for r in base_results])
    mean_ttr_base = np.mean([float(r["TTR (steps)"]) if not str(r["TTR (steps)"]).startswith(">") else float(r["TTR (steps)"][1:]) for r in base_results])
    
    # Build comparison df
    comp_df = pd.DataFrame([
        {
            "Metric": "Worst Victim Survival Rate (Worst VSR)",
            f"Active Config ({mode})": f"{worst_vsr_active:.3f}",
            "Greedy Router Baseline": f"{worst_vsr_base:.3f}",
            "Difference": f"{worst_vsr_active - worst_vsr_base:+.3f} (Lower risk!)" if (worst_vsr_active > worst_vsr_base) else f"{worst_vsr_active - worst_vsr_base:+.3f}"
        },
        {
            "Metric": "Mean Victim Survival Rate (Mean VSR)",
            f"Active Config ({mode})": f"{mean_vsr_active:.3f}",
            "Greedy Router Baseline": f"{mean_vsr_base:.3f}",
            "Difference": f"{mean_vsr_active - mean_vsr_base:+.3f}"
        },
        {
            "Metric": "Mean Time to Rescue (Mean TTR - steps)",
            f"Active Config ({mode})": f"{mean_ttr_active:.1f}",
            "Greedy Router Baseline": f"{mean_ttr_base:.1f}",
            "Difference": f"{mean_ttr_active - mean_ttr_base:+.1f} steps (Faster!)" if (mean_ttr_active < mean_ttr_base) else f"{mean_ttr_active - mean_ttr_base:+.1f} steps"
        },
        {
            "Metric": "Total Branching Events (AES)",
            f"Active Config ({mode})": f"{branches}",
            "Greedy Router Baseline": "0 (No active sensing)",
            "Difference": f"+{branches}"
        }
    ])
    
    st.table(comp_df)
    
    # Side-by-side Victim comparison details
    comp_vics = []
    for idx in range(len(results)):
        res_a = results[idx]
        res_b = base_results[idx]
        comp_vics.append({
            "Victim": res_a["Victim"],
            f"Rescued ({mode})": res_a["Rescued"],
            "Rescued (Greedy)": res_b["Rescued"],
            f"TTR ({mode})": res_a["TTR (steps)"],
            "TTR (Greedy)": res_b["TTR (steps)"],
            f"VSR ({mode})": res_a["VSR"],
            "VSR (Greedy)": res_b["VSR"]
        })
    st.dataframe(pd.DataFrame(comp_vics), use_container_width=True, hide_index=True)

with tab3:
    st.markdown("### 🖼️ Evidential Visual Playground")
    st.markdown("Upload any maritime image crop (e.g. swimmer, boat, debris) to see real-time evidential classification, expected risk calculation, and epistemic uncertainty prediction.")

    uploaded_file = st.file_uploader("Choose a maritime image crop...", type=["jpg", "png", "jpeg", "webp"])
    
    if uploaded_file is not None:
        from PIL import Image, ImageFilter
        import torch
        import torchvision.transforms as transforms
        from src.models import EvidentialCNNClassifier
        
        # Load and display uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        
        # Sidelining options to simulate environment
        col_img_1, col_img_2 = st.columns([1, 1])
        with col_img_1:
            st.markdown("**Original Uploaded Image**")
            st.image(image, use_container_width=True)
            
        with col_img_2:
            st.markdown("**Simulated Environmental Deterioration**")
            sim_occlusion = st.checkbox("Simulate Wave Occlusion (Gaussian Blur)", value=False)
            sim_dist = st.slider("Simulated UAV Distance (m)", 5.0, 100.0, 15.0, 1.0)
            
            # Apply same pipeline transformation as simulator
            img_work = image.copy()
            if sim_occlusion:
                img_work = img_work.filter(ImageFilter.GaussianBlur(12.0))
            if sim_dist > 25.0:
                # Downsample
                img_work = img_work.resize((8, 8)).resize((64, 64))
            else:
                img_work = img_work.resize((64, 64))
                
            st.image(img_work, caption="Processed Network Input (64x64)", use_container_width=True)
            
        # Run Inference
        weights_path = "models/edl_weights.pth"
        if not os.path.exists(weights_path):
            st.error("Model weights not found at `models/edl_weights.pth`. Please train the model first.")
        else:
            try:
                # Cache model to avoid reloading on every rerun
                @st.cache_resource
                def load_inference_model():
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model_instance = EvidentialCNNClassifier(num_classes=5)
                    model_instance.load_state_dict(torch.load(weights_path, map_location=device))
                    model_instance.to(device)
                    model_instance.eval()
                    return model_instance, device
                    
                model, device = load_inference_model()
                
                # Preprocess image
                t = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                img_t = t(img_work).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    evidence = model(img_t).cpu().numpy()[0]
                    
                alpha = evidence + 1.0
                S_sum = np.sum(alpha)
                beliefs_raw = evidence / S_sum
                u_val = 5.0 / S_sum
                probs_raw = alpha / S_sum
                
                # Risk Weights: swimmer, floater, boat, life jacket, buoy
                classes_labels_5 = ["Swimmer", "Floater", "Boat", "Life Jacket", "Buoy"]
                risk_weights = np.array([1.0, 0.4, 0.2, 0.1, 0.05])
                expected_risk = float(np.sum(probs_raw * risk_weights))
                
                # Metrics output
                col_res1, col_res2, col_res3 = st.columns(3)
                with col_res1:
                    st.metric("🔬 Epistemic Uncertainty (u)", f"{u_val:.3f}")
                with col_res2:
                    st.metric("⚠️ Expected Posture Risk (E[R])", f"{expected_risk:.3f}")
                with col_res3:
                    # Predicted Class
                    pred_idx = int(np.argmax(probs_raw))
                    st.metric("🏷️ Predicted Category", classes_labels_5[pred_idx])
                    
                # Visual Evidence distribution bar chart
                st.markdown("#### 📊 Dirichlet Evidence Vector (Alpha)")
                
                evidence_df = pd.DataFrame({
                    "Category": classes_labels_5,
                    "Raw Evidence (e)": [float(e) for e in evidence],
                    "Belief Mass (b)": [float(b) for b in beliefs_raw]
                })
                
                st.dataframe(evidence_df, use_container_width=True, hide_index=True)
                
                # Graph of evidence
                fig_play = go.Figure()
                fig_play.add_trace(go.Bar(
                    x=classes_labels_5,
                    y=beliefs_raw,
                    name="Belief Mass",
                    marker_color="#00d2ff"
                ))
                fig_play.update_layout(
                    title="Estimated Class Probabilities (Belief)",
                    yaxis=dict(range=[0, 1], gridcolor=grid_color, tickfont=dict(color=text_color_plotly)),
                    xaxis=dict(tickfont=dict(color=text_color_plotly)),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=text_color_plotly)
                )
                st.plotly_chart(fig_play, use_container_width=True)
                
            except Exception as e:
                st.error(f"Inference failed: {e}")

# Handle autoplay rerun at the end of the script after rendering is complete
if autoplay:
    time.sleep(speed_factor)
    st.session_state.step_counter = (st.session_state.step_counter + 1) % max_steps
    st.rerun()

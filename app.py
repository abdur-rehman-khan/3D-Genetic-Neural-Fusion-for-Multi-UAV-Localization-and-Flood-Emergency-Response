"""GenNFRescue - Streamlit dashboard.  Run:  python -m streamlit run app.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st

from gennf.rssi_model import RSSIModel
from gennf.simulation import SimConfig, Simulation
from gennf import visualization as viz

st.set_page_config(page_title="GenNFRescue", page_icon="🛸", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
:root {
    --app-bg: #17191D;
    --panel-bg: #22252A;
    --panel-inner: #292D33;
    --control-bg: #30353C;
    --border: #414750;
    --text: #E8EAED;
    --muted: #B0B7C3;
    --blue: #3B82F6;
    --blue-hover: #60A5FA;
    --yellow: #FACC15;
}
.stApp, [data-testid="stAppViewContainer"], .main { background: var(--app-bg) !important; color: var(--text) !important; }
header[data-testid="stHeader"] { background: var(--app-bg) !important; }
.block-container { max-width: 100% !important; padding: 0.65rem 0.75rem 1rem !important; }
div[data-testid="stHorizontalBlock"] { gap: 0.65rem !important; align-items: stretch !important; }
div[data-testid="column"] { min-width: 0 !important; }
div[data-testid="stVerticalBlockBorderWrapper"] { background: var(--panel-bg) !important; border: 1px solid var(--border) !important; border-radius: 6px !important; padding: 0.65rem !important; }
.stApp p, .stApp li, .stApp label, .stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stMetricLabel"] { color: var(--muted); }
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp strong { color: var(--text) !important; }
.stApp h3 { font-size: 1rem !important; font-weight: 650 !important; margin: 0.25rem 0 0.7rem !important; }
div[data-testid="stVerticalBlockBorderWrapper"] h3 { padding-bottom: 0.45rem; border-bottom: 1px solid var(--border); }
.stApp input, .stApp textarea, .stApp div[data-baseweb="select"] > div { background-color: var(--control-bg) !important; color: var(--text) !important; border-color: var(--border) !important; }
.stApp [data-baseweb="select"] *, .stApp [data-testid="stNumberInput"] button { color: var(--text) !important; }
.stApp [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] { background: var(--blue) !important; border-color: var(--blue) !important; }
.stApp [data-testid="stSlider"] label { color: var(--muted) !important; }
.stApp button[kind="primary"] { background: #F55151 !important; color: white !important; border: 1px solid #f59e0b !important; border-radius: 5px !important; font-weight: 600 !important; }
.stApp button[kind="primary"]:hover { background: #AD2323 !important; }
.stApp button[kind="secondary"] { background: var(--control-bg) !important; color: var(--text) !important; border: 6px solid var(--border) !important; }
div[data-testid="stMetric"] { background: var(--panel-inner); border: 1px solid var(--border); border-radius: 5px; padding: 0.55rem !important; }
div[data-testid="stMetricValue"] { color: var(--text) !important; font-size: 1.25rem !important; }
div[data-testid="column"]:first-child { background:#000000 !important; }
div[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 5px !important; overflow: hidden; }
* { scrollbar-width: thin; scrollbar-color: #515966 var(--app-bg); }
*::-webkit-scrollbar { width: 8px; height: 8px; }
*::-webkit-scrollbar-track { background: var(--app-bg); }
*::-webkit-scrollbar-thumb { background: #515966; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align:center; margin-top:10px; margin-bottom:15px;'>3D Genetic Neural Fusion for Multi UAV Localization and Flood Emergency Response</h2>", unsafe_allow_html=True)
left, center, right = st.columns([0.23, 0.47, 0.30], gap="small", vertical_alignment="top")

with left:
    with st.container(border=True):
        st.markdown("## Attributes")
        seed = st.number_input("Random seed", 0, 10_000, 42)
        n_uavs = st.slider("UAV nodes", 4, 20, 10)
        n_victims = st.slider("Victims", 1, 12, 6)
        formation = st.selectbox("UAV deployment", ["two_layer", "ring", "grid", "random"])
        max_time = st.slider("Max sim time (s)", 60, 600, 300, 30)
    with st.container(border=True):
        st.markdown("## RSSI")
        sigma = st.slider("Noise σ (dB)", 0.0, 8.0, 1.0, 0.5)
        n_exp = st.slider("Path-loss n", 1.5, 4.0, 2.7, 0.1)
        p0 = st.slider("P0 (dBm)", -60, -20, -40)
        comm = st.slider("Comm range (m)", 60, 300, 150, 10)
    with st.container(border=True):
        st.markdown("## GenNF")
        geometry = st.selectbox("Geometric stage", ["trilateration", "pairwise"])
        use_ga = st.checkbox("Use GA init", True)
        hidden_str = st.text_input("ANN hidden", "64,32")
        ga_pop = st.slider("GA population", 20, 500, 200, 10)
        ga_gen = st.slider("GA generations", 10, 300, 100, 10)
        ga_mr = st.slider("GA mutation rate", 0.01, 0.5, 0.1, 0.01)
        ga_ms = st.slider("GA mutation sigma", 0.01, 0.5, 0.05, 0.01)
    with st.container(border=True):
        st.markdown("## Rescue")
        thr = st.slider("Success thr (m)", 5, 50, 20)
        drop = st.slider("Drop radius (m)", 5, 60, 30)
        frame_ms = st.slider("Frame (ms)", 20, 400, 80, 10)
        run = st.button("▶ Run / Reset", type="primary", use_container_width=True)

hidden = tuple(int(x.strip()) for x in hidden_str.split(",") if x.strip())
cfg = SimConfig(seed=int(seed), n_uavs=n_uavs, n_victims=n_victims, formation=formation, max_time=float(max_time),
                rssi=RSSIModel(p0, n_exp, 1.0, sigma, float(comm)), success_threshold_m=float(thr),
                delivery_radius_m=float(drop), use_ga=use_ga, geometry=geometry,
                hidden=hidden, ga_population=ga_pop, ga_generations=ga_gen,
                ga_mutation_rate=ga_mr, ga_mutation_sigma=ga_ms)

if run or "sim" not in st.session_state:
    with st.spinner("Training GenNF and running mission…"):
        st.session_state.sim = Simulation(cfg).run()
        st.session_state.fig = viz.build_figure(st.session_state.sim, frame_ms=frame_ms)
sim = st.session_state.sim

with center:
    st.markdown("## 3D Map")
    last = len(sim.history) - 1
    k = st.slider("Step", 0, last, last)
    m = sim.metrics(k)
    c = st.columns(5)
    c[0].metric("Time", f"{m['sim_time_s']:.1f}s"); c[1].metric("UAVs", f"{m['active_uavs']}/{len(sim.net.uavs)}")
    c[2].metric("Detected", m["victims_detected"]); c[3].metric("Localized", m["victims_localized"])
    c[4].metric("Missions", m["missions_completed"])
    c2 = st.columns(4)
    c2[0].metric("Mean err", f"{m['mean_error_m']:.1f}m"); c2[1].metric("RMSE", f"{m['rmse_3d_m']:.1f}m")
    c2[2].metric("MSE", f"{m['mse_3d_m']:.1f}m²"); c2[3].metric("MAE", f"{m['mae_axis_m']:.1f}m")
    st.plotly_chart(st.session_state.fig, use_container_width=True)

with right:
    st.markdown("## Analytics")
    st.plotly_chart(viz.style_chart(viz.chart_true_vs_est(sim, k), 230), use_container_width=True, key="chart_true_est")
    st.plotly_chart(viz.style_chart(viz.chart_errors(sim, k), 190), use_container_width=True, key="chart_errors")
    st.plotly_chart(viz.style_chart(viz.chart_uav_status(sim, k), 180), use_container_width=True, key="chart_uav_status")
    st.plotly_chart(viz.style_chart(viz.chart_missions(sim, k), 180), use_container_width=True, key="chart_missions")
    vsel = st.selectbox("Victim", range(len(sim.victims)), format_func=lambda i: sim.victims[i].name)
    st.plotly_chart(viz.style_chart(viz.chart_rssi(sim, vsel), 210), use_container_width=True, key="chart_rssi")
    st.markdown("### Data")
    st.dataframe(sim.victims_table(), use_container_width=True, height=200)
    st.dataframe(sim.uav_table(), use_container_width=True, height=200)

# ============================================================================
# PROJECT: AI-Powered Sports Analytics Platform (Pro Glass Edition)
# FILE: app.py
# DESCRIPTION: Advanced Green Transparent Glassmorphic Theme Dashboard with Cleaned Layout.
# ============================================================================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tempfile
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras

from utils.cv_engine import SportsCVEngine

# -----------------------------------------------------------------------------
# Page setup and visual theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Sports Analytics Platform",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --green-primary: #10b981;
        --green-bright: #34d399;
        --green-mint: #6ee7b7;
        --dark-emerald: #064e3b;
        --app-bg-start: #022c22;
        --app-bg-end: #021f18;
        --panel-bg: rgba(6, 78, 59, 0.45);
        --border-color: rgba(16, 185, 129, 0.35);
        --text-main: #ecfdf5;
        --text-muted: #a7f3d0;
    }
    
    .stApp {
        background: radial-gradient(circle at 15% 15%, rgba(16, 185, 129, 0.22) 0%, #021f18 75%) !important;
        color: var(--text-main);
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    [data-testid="stHeader"] { 
        background: rgba(2, 44, 34, 0.65); 
        backdrop-filter: blur(14px); 
        -webkit-backdrop-filter: blur(14px);
    }
    
    [data-testid="stSidebar"] {
        background: rgba(2, 31, 24, 0.85) !important;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-right: 1px solid var(--border-color);
    }
    [data-testid="stSidebar"] * { 
        color: var(--text-main) !important; 
    }
    
    .hero {
        padding: 30px 34px;
        border-radius: 20px;
        border: 1px solid var(--border-color);
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.65) 0%, rgba(2, 44, 34, 0.80) 100%);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        box-shadow: 0 20px 45px rgba(0, 0, 0, 0.45), 0 0 25px rgba(16, 185, 129, 0.15);
        margin: 6px 0 24px;
    }
    .hero h1 { 
        color: var(--green-bright); 
        font-size: clamp(1.7rem, 3.2vw, 2.5rem); 
        margin: 0 0 8px; 
        letter-spacing: .02em; 
        text-shadow: 0 0 18px rgba(52, 211, 153, 0.35);
    }
    .hero p { color: var(--text-muted); margin: 0; font-size: 1.02rem; }
    .eyebrow { color: var(--green-mint); font-size: .8rem; font-weight: 700; letter-spacing: .15em; text-transform: uppercase; margin-bottom: 6px; }
    
    .panel {
        border: 1px solid var(--border-color);
        border-radius: 18px;
        background: var(--panel-bg);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        padding: 26px;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.4);
        margin: 6px 0 18px;
    }
    
    .info-box, .warning-box {
        padding: 14px 16px;
        border-radius: 12px;
        margin: 10px 0 16px;
        font-size: .94rem;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
    }
    .info-box { 
        color: #d1fae5; 
        background: rgba(16, 185, 129, 0.14); 
        border: 1px solid rgba(16, 185, 129, 0.45); 
        border-left: 4px solid var(--green-bright); 
    }
    .warning-box { 
        color: #fef3c7; 
        background: rgba(245, 158, 11, 0.14); 
        border: 1px solid rgba(245, 158, 11, 0.45); 
        border-left: 4px solid #f59e0b; 
    }
    
    div[data-testid="stFileUploader"] {
        background: rgba(2, 44, 34, 0.55) !important;
        border: 2px dashed var(--green-primary) !important;
        border-radius: 16px !important;
        padding: 16px !important;
        backdrop-filter: blur(12px);
    }
    div[data-testid="stFileUploader"] section {
        background-color: transparent !important;
    }
    
    .metric-box {
        background: rgba(2, 31, 24, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--border-color);
        border-left: 4px solid var(--green-bright);
        border-radius: 14px;
        padding: 18px 20px;
        margin: 12px 0;
        color: var(--text-main);
        line-height: 1.8;
    }
    
    .stButton > button, .stDownloadButton > button {
        border: none;
        border-radius: 12px;
        min-height: 3rem;
        color: #022c22 !important;
        background: linear-gradient(135deg, var(--green-bright) 0%, var(--green-primary) 100%) !important;
        font-weight: 800;
        transition: transform .2s ease, box-shadow .2s ease, filter .2s ease;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.45);
        width: 100%;
        letter-spacing: .02em;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        color: #022c22 !important;
        filter: brightness(1.15);
        transform: translateY(-2.5px);
        box-shadow: 0 8px 26px rgba(52, 211, 153, 0.65);
    }
    
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
        background-color: rgba(2, 44, 34, 0.8) !important;
        color: var(--text-main) !important;
        border-radius: 10px;
        border: 1px solid var(--border-color) !important;
        backdrop-filter: blur(12px);
    }
    div[data-baseweb="input"] input, div[data-baseweb="select"] span {
        color: var(--text-main) !important;
    }
    div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] { 
        background: var(--green-bright); 
        box-shadow: 0 0 12px var(--green-bright);
    }
    
    [data-testid="stMetric"] {
        background: var(--panel-bg);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }
    [data-testid="stMetricValue"] { color: var(--green-bright) !important; text-shadow: 0 0 12px rgba(52, 211, 153, 0.3); }
    
    h1, h2, h3, h4, p, label { color: var(--text-main); }
    .footer { text-align: center; color: var(--text-muted); font-size: .86rem; padding: 20px 0 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner="Loading computer-vision engine…")
def load_cv_engine():
    return SportsCVEngine()

@st.cache_resource(show_spinner="Loading available deep-learning models…")
def load_dl_models():
    models = {}
    assets = {
        "perf_model": ("models/performance_model.keras", keras.models.load_model),
        "perf_scaler": ("models/scaler_performance.pkl", joblib.load),
        "match_model": ("models/match_outcome_model.keras", keras.models.load_model),
        "match_scaler": ("models/scaler_match.pkl", joblib.load),
        "lstm_model": ("models/forecasting_lstm.keras", keras.models.load_model),
        "lstm_scaler": ("models/scaler_lstm.pkl", joblib.load),
    }
    for key, (asset_path, loader) in assets.items():
        try:
            if Path(asset_path).is_file():
                models[key] = loader(asset_path)
        except Exception as exc:
            models[f"{key}_error"] = str(exc)
    return models

try:
    cv_engine = load_cv_engine()
    cv_engine_error = None
except Exception as exc:
    cv_engine = None
    cv_engine_error = str(exc)

dl_models = load_dl_models()

def init_state():
    defaults = {
        "final_tracked_metrics": {},
        "extracted_stamina": 82.0,
        "extracted_pass": 88.0,
        "extracted_dist": 11.2,
        "extracted_speed": 31.0,
        "extracted_actions": 25,
        "transfer_success": False,
        "video_bytes": None,
        "video_name": None,
        "run_live": False,
        "perf_result": None,
        "match_result": None,
        "forecast_result": None,
        "last_uploaded_name": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_state()

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">⚡ Computer Vision · Neural Networks · Match Intelligence</div>
      <h1>AI Sports Analytics Platform</h1>
      <p>Real-time player tracking, telemetry sync, performance evaluation, and predictive insights in a professional workspace.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

modules = [
    "📹 CV Tracking & Telemetry",
    "📊 Performance Evaluator (ANN)",
    "🏟️ Match Outcome Predictor (ANN)",
    "📈 Rating Forecasting (LSTM)",
    "📂 Dataset Explorer",
]

# Updated title in sidebar as requested
st.sidebar.markdown("## ⚡ Sports Analytics & Performance Platform")
st.sidebar.caption("Select your analytics module")
selected_module = st.sidebar.radio("Analytics module", modules, label_visibility="collapsed")
st.sidebar.markdown("---")
if cv_engine_error:
    st.sidebar.warning("⚠️ CV engine unavailable")
else:
    st.sidebar.success("🟢 Platform ready & active")

# -----------------------------------------------------------------------------
# Module 1: Video tracking and telemetry
# -----------------------------------------------------------------------------
if selected_module == "📹 CV Tracking & Telemetry":
    st.subheader("🎥 Player Detection & Live Telemetry Stream")
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### 📤 Upload Match Footage")
        uploaded_video = st.file_uploader(
            "Choose an MP4, AVI, or MOV file",
            type=["mp4", "avi", "mov"],
            key="match_video_uploader",
            help="Your video is processed securely in this session.",
        )
        
        if uploaded_video is not None:
            if st.session_state.last_uploaded_name != uploaded_video.name:
                st.session_state.video_bytes = uploaded_video.getvalue()
                st.session_state.video_name = uploaded_video.name
                st.session_state.last_uploaded_name = uploaded_video.name
                st.session_state.final_tracked_metrics = {}
                st.session_state.transfer_success = False
                st.session_state.run_live = False

        video_bytes = st.session_state.video_bytes
        video_name = st.session_state.video_name
        if video_bytes:
            st.video(video_bytes)
            st.caption(f"📁 Selected file: {video_name}")
            action_left, action_right = st.columns(2)
            with action_left:
                start_detection = st.button("▶ Start Analysis")
            with action_right:
                stop_detection = st.button("⏹ Stop Analysis")
            
            if start_detection:
                st.session_state.run_live = True
            if stop_detection:
                st.session_state.run_live = False
        else:
            st.markdown('<div class="warning-box">⚠️ Please upload a match clip first. Supported formats: MP4, AVI, MOV.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### 🔍 Detection & Telemetry Output")
        
        if video_bytes and st.session_state.run_live:
            if cv_engine is None:
                st.error("Tracking cannot run because the CV engine failed to load.")
            else:
                cap = None
                frame_placeholder = st.empty()
                progress_bar = st.progress(0.0)
                status = st.empty()
                player_stats = {}
                processed_count = 0
                last_annotated = None
                frame_skip = 2
                try:
                    with tempfile.NamedTemporaryFile(suffix=Path(video_name or "clip.mp4").suffix or ".mp4", delete=False) as temp_video:
                        temp_video.write(video_bytes)
                        video_path = temp_video.name
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        st.error("OpenCV could not open this video stream.")
                    else:
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
                        frame_idx = 0
                        
                        while cap.isOpened() and st.session_state.run_live:
                            ok, frame = cap.read()
                            if not ok:
                                break
                            frame_idx += 1
                            if frame_idx % frame_skip == 0 or last_annotated is None:
                                try:
                                    annotated, active_count = cv_engine.process_frame_and_track(frame, frame_idx, player_stats)
                                    last_annotated = annotated if annotated is not None else frame
                                    processed_count += 1
                                except Exception as exc:
                                    st.error(f"Frame processing error: {exc}")
                                    break
                            display_frame = last_annotated if last_annotated is not None else frame
                            frame_placeholder.image(display_frame, channels="BGR", use_container_width=True)
                            if total_frames:
                                progress_bar.progress(min(frame_idx / total_frames, 1.0))
                                status.caption(f"Processing frame {frame_idx:,} of {total_frames:,}")
                            else:
                                status.caption(f"Processing frame {frame_idx:,}")
                        
                        if cap is not None and cap.isOpened():
                            cap.release()
                        cap = None
                        
                        if player_stats:
                            final_metrics = cv_engine.calculate_final_player_metrics(player_stats)
                            st.session_state.final_tracked_metrics = final_metrics or {}
                            st.session_state.run_live = False
                            st.rerun()
                except Exception as exc:
                    st.error(f"Video analysis failed: {exc}")
                finally:
                    if cap is not None:
                        cap.release()
                    if "video_path" in locals() and os.path.exists(video_path):
                        try:
                            os.remove(video_path)
                        except OSError:
                            pass
                    st.session_state.run_live = False
        else:
            if not st.session_state.final_tracked_metrics:
                st.markdown('<div class="info-box">💡 Upload a video and click <b>Start Analysis</b> to extract player metrics in real time.</div>', unsafe_allow_html=True)

        detected_players = list(st.session_state.final_tracked_metrics.keys())
        if detected_players:
            st.success(f"✅ Analysis complete! Detected {len(detected_players)} player(s).")
            st.markdown("---")
            st.markdown('<div class="info-box">✨ Select a player below to inspect metrics and sync them directly with Module 2.</div>', unsafe_allow_html=True)
            
            selected_player = st.selectbox("Choose detected player", detected_players, key="tracked_player_select")
            p_info = st.session_state.final_tracked_metrics[selected_player]
            
            st.markdown(
                f"""<div class="metric-box">📊 <b>{selected_player} Telemetry Profile:</b><br>
                • Stamina Index: <b>{p_info.get('stamina', 'N/A')}</b> | Pass Accuracy: <b>{p_info.get('pass_accuracy', 'N/A')}%</b><br>
                • Distance Covered: <b>{p_info.get('distance', 'N/A')} km</b> | Max Velocity: <b>{p_info.get('speed', 'N/A')} km/h</b><br>
                • Key Actions: <b>{p_info.get('key_actions', 'N/A')}</b></div>""",
                unsafe_allow_html=True,
            )
            
            metric_df = pd.DataFrame({
                "Metric": ["Stamina", "Pass Acc", "Distance", "Speed", "Actions"],
                "Value": [
                    float(p_info.get('stamina', 0)),
                    float(p_info.get('pass_accuracy', 0)),
                    float(p_info.get('distance', 0)) * 5,
                    float(p_info.get('speed', 0)),
                    float(p_info.get('key_actions', 0))
                ]
            }).set_index("Metric")
            st.bar_chart(metric_df, color="#34d399")
            
            if st.button("🚀 Sync Telemetry to Performance Evaluator", key="transfer_telemetry"):
                try:
                    st.session_state.extracted_stamina = float(p_info["stamina"])
                    st.session_state.extracted_pass = float(p_info["pass_accuracy"])
                    st.session_state.extracted_dist = float(p_info["distance"])
                    st.session_state.extracted_speed = float(p_info["speed"])
                    st.session_state.extracted_actions = int(p_info["key_actions"])
                    
                    st.session_state.transfer_success = True
                    st.success(f"✅ {selected_player} telemetry data synced successfully! Switch to **Performance Evaluator (ANN)** in the sidebar.")
                except (KeyError, TypeError, ValueError) as exc:
                    st.error(f"Cannot sync telemetry: {exc}")
        else:
            if not st.session_state.run_live:
                st.markdown(
                    """<div class="warning-box">
                    🔒 <b>Player selection locked:</b> Run video analysis first to unlock telemetry selection.
                    </div>""",
                    unsafe_allow_html=True
                )
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Module 2: Performance ANN
# -----------------------------------------------------------------------------
elif selected_module == "📊 Performance Evaluator (ANN)":
    st.subheader("📊 Player Performance Evaluator (ANN)")
    st.caption("Estimate player performance ratings using deep neural networks or synced match telemetry.")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    if st.session_state.transfer_success:
        st.markdown('<div class="info-box">🔄 Synced live telemetry data loaded successfully from CV Tracking module!</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">💡 Tip: You can also adjust sliders manually to test custom performance scenarios.</div>', unsafe_allow_html=True)

    stamina_default = float(np.clip(st.session_state.extracted_stamina, 50.0, 100.0))
    pass_default = float(np.clip(st.session_state.extracted_pass, 60.0, 98.0))
    dist_default = float(np.clip(st.session_state.extracted_dist, 4.0, 20.0))
    speed_default = float(np.clip(st.session_state.extracted_speed, 18.0, 50.0))
    actions_default = int(np.clip(st.session_state.extracted_actions, 5, 50))
    
    c1, c2 = st.columns(2, gap="large")
    with c1:
        stamina = st.slider("Stamina Score", 50.0, 100.0, stamina_default, key="perf_stamina")
        pass_acc = st.slider("Pass Accuracy (%)", 60.0, 98.0, pass_default, key="perf_pass")
        distance = st.number_input("Distance Covered (km)", 4.0, 20.0, dist_default, step=0.1, key="perf_distance")
    with c2:
        speed = st.number_input("Max Speed (km/h)", 18.0, 50.0, speed_default, step=0.5, key="perf_speed")
        actions = st.slider("Key Actions Count", 5, 50, actions_default, key="perf_actions")

    if st.button("✨ Evaluate Performance Rating", key="evaluate_performance"):
        try:
            if "perf_model" in dl_models and "perf_scaler" in dl_models:
                input_data = np.array([[stamina, pass_acc, distance, speed, actions]], dtype=float)
                scaled = dl_models["perf_scaler"].transform(input_data)
                pred = dl_models["perf_model"].predict(scaled, verbose=0)
                rating = float(np.asarray(pred).reshape(-1)[0])
                model_label = "Trained ANN Model"
            else:
                rating = (stamina * 0.25 + pass_acc * 0.35 + distance * 2.5 + speed * 0.8 + actions * 0.4) / 10.0
                model_label = "Interactive Heuristic Engine"
            rating = round(float(np.clip(rating, 1.0, 10.0)), 2)
            st.session_state.perf_result = {"rating": rating, "model": model_label}
        except Exception as exc:
            st.error(f"Performance prediction failed: {exc}")
            
    if st.session_state.perf_result:
        result = st.session_state.perf_result
        st.metric("Predicted Player Performance Rating", f"{result['rating']} / 10")
        st.caption(f"Engine source: {result['model']}")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Module 3: Match outcome ANN
# -----------------------------------------------------------------------------
elif selected_module == "🏟️ Match Outcome Predictor (ANN)":
    st.subheader("🏟️ Match Outcome Predictor")
    st.caption("Forecast match outcomes and win probabilities based on team-level telemetry.")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        possession = st.slider("Ball Possession (%)", 30.0, 70.0, 55.0, key="match_possession")
        shots = st.number_input("Shots on Target", 1, 15, 8, key="match_shots")
        pass_success = st.slider("Pass Success (%)", 55.0, 92.0, 78.0, key="match_pass")
    with c2:
        fouls = st.number_input("Fouls Committed", 2, 20, 8, key="match_fouls")
        defense = st.slider("Defense Rating", 50.0, 95.0, 75.0, key="match_defense")

    if st.button("🔮 Predict Match Result", key="predict_match"):
        try:
            features = np.array([[possession, shots, pass_success, fouls, defense]], dtype=float)
            if "match_model" in dl_models and "match_scaler" in dl_models:
                scaled = dl_models["match_scaler"].transform(features)
                raw = np.asarray(dl_models["match_model"].predict(scaled, verbose=0)).reshape(-1)
                win_probability = float(np.clip(raw[-1] if raw.size >= 2 else raw[0], 0.0, 1.0))
                source = "Trained ANN Model"
            else:
                score = (possession * 0.3) + (shots * 3.5) + (pass_success * 0.2) - (fouls * 1.2) + (defense * 0.25)
                win_probability = float(1 / (1 + np.exp(-np.clip((score - 55) / 15, -30, 30))))
                source = "Interactive Heuristic Engine"
            outcome = "Win 🏆" if win_probability >= 0.5 else "Loss / Draw ⚠️"
            st.session_state.match_result = {"outcome": outcome, "probability": win_probability, "source": source}
        except Exception as exc:
            st.error(f"Match prediction failed: {exc}")
            
    if st.session_state.match_result:
        result = st.session_state.match_result
        st.metric("Predicted Outcome", result["outcome"], f"Confidence Level: {result['probability']:.1%}")
        st.progress(result['probability'])
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Module 4: LSTM rating forecast
# -----------------------------------------------------------------------------
elif selected_module == "📈 Rating Forecasting (LSTM)":
    st.subheader("📈 Next-Match Rating Forecast (LSTM)")
    st.caption("Input chronological historical ratings to forecast upcoming match performance.")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    cols = st.columns(5)
    defaults = [6.5, 7.0, 6.8, 7.5, 8.0]
    ratings = [cols[i].number_input(f"Match {i + 1}", 1.0, 10.0, defaults[i], step=0.1, key=f"lstm_rating_{i}") for i in range(5)]

    hist_df = pd.DataFrame({
        "Match": [f"Match {i+1}" for i in range(5)],
        "Rating": ratings
    }).set_index("Match")
    st.line_chart(hist_df, color="#34d399")

    if st.button("📈 Forecast Upcoming Rating", key="forecast_rating"):
        try:
            sequence = np.asarray(ratings, dtype=float).reshape(-1, 1)
            if "lstm_model" in dl_models:
                scaler = dl_models.get("lstm_scaler")
                model = dl_models["lstm_model"]
                prepared = scaler.transform(sequence) if scaler is not None else sequence
                prepared = prepared.reshape(1, 5, 1)
                raw_prediction = np.asarray(model.predict(prepared, verbose=0)).reshape(-1)
                prediction = float(raw_prediction[0])
                if scaler is not None and hasattr(scaler, "inverse_transform"):
                    prediction = float(scaler.inverse_transform(np.array([[prediction]])).reshape(-1)[0])
                source = "Trained LSTM Model"
            else:
                prediction = float(np.mean(ratings) + 0.2 * (ratings[-1] - ratings[0]))
                source = "Trend Extrapolation Engine"
            prediction = round(float(np.clip(prediction, 1.0, 10.0)), 2)
            st.session_state.forecast_result = {"rating": prediction, "source": source}
        except Exception as exc:
            st.error(f"Forecast failed: {exc}")
            
    if st.session_state.forecast_result:
        result = st.session_state.forecast_result
        st.metric("Forecasted Rating for Next Match", f"{result['rating']} / 10")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Module 5: Dataset explorer
# -----------------------------------------------------------------------------
elif selected_module == "📂 Dataset Explorer":
    st.subheader("📂 Training Dataset Explorer")
    st.caption("Inspect raw datasets and telemetry logs used across your models.")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    dataset_paths = [
        "data/player_performance_data.csv",
        "data/match_outcome_data.csv",
        "data/player_timeseries_data.csv",
    ]
    selected_dataset = st.selectbox("Select dataset", dataset_paths, key="dataset_choice")
    if os.path.isfile(selected_dataset):
        try:
            df = pd.read_csv(selected_dataset)
            st.caption(f"📊 {len(df):,} total rows · {len(df.columns)} features")
            st.dataframe(df.head(100), use_container_width=True, hide_index=True)
            
            with st.expander("📈 View Dataset Statistical Summary"):
                st.write(df.describe())
        except Exception as exc:
            st.error(f"Could not read dataset: {exc}")
    else:
        st.warning(f"⚠️ Dataset not found: {selected_dataset}.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    '<div class="footer">AI Sports Analytics Platform · Pro Green Transparent Glass Edition · Streamlit, OpenCV & TensorFlow</div>',
    unsafe_allow_html=True,
)
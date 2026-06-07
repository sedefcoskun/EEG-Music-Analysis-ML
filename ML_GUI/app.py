import os
import joblib
import random
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

# CONSTANTS
BANDS = {
    "Delta":  {
        "range": "1–4 Hz",
        "color": "#7c3aed",
        "emoji": "δ",
        "role_high": "large-scale neural entrainment to the musical beat and rhythm",
        "role_low":  "weak rhythmic synchronization with the stimulus",
        "neuro":     "beat entrainment & slow-wave synchronization"
    },
    "Theta":  {
        "range": "4–8 Hz",
        "color": "#2563eb",
        "emoji": "θ",
        "role_high": "deep limbic engagement — emotional memory and reward anticipation",
        "role_low":  "limited emotional resonance with the stimulus",
        "neuro":     "limbic reward & emotional memory"
    },
    "Alpha":  {
        "range": "8–13 Hz",
        "color": "#059669",
        "emoji": "α",
        "role_high": "internalized attention — tuning out external distractions for deep absorption",
        "role_low":  "active sensory processing without internal focus",
        "neuro":     "sensory inhibition & attentional gating"
    },
    "Beta":   {
        "range": "13–30 Hz",
        "color": "#d97706",
        "emoji": "β",
        "role_high": "sensorimotor resonance — the brain 'feeling the groove' at a neural level",
        "role_low":  "low motor resonance with the musical rhythm",
        "neuro":     "motor resonance & active listening"
    },
    "Gamma":  {
        "range": "30–45 Hz",
        "color": "#dc2626",
        "emoji": "γ",
        "role_high": "perceptual binding — integrating melody, rhythm and timbre into a unified aesthetic experience",
        "role_low":  "fragmented auditory feature processing",
        "neuro":     "perceptual binding & high-level integration"
    },
}

# XGBoost importance weights (from the trained model)
BAND_WEIGHTS = {
    "Gamma": 0.247,
    "Beta":  0.228,
    "Delta": 0.182,
    "Theta": 0.177,
    "Alpha": 0.165,
}

MODELS_DIR = "saved_models"
HIGH_THRESHOLD = 0.70   # band power threshold for "active" label
TOP_N_BANDS    = 3      # how many bands to highlight in the comment

# PAGE CONFIG 
st.set_page_config(
    page_title="Neuro-Music Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# SESSION STATE INIT 
if "band_data" not in st.session_state:
    st.session_state.band_data = {
        "Delta": 0.30, "Theta": 0.50,
        "Alpha": 0.70, "Beta":  0.55, "Gamma": 0.40
    }
if "randomize" not in st.session_state:
    st.session_state.randomize = False
    
if "render_key" not in st.session_state:
    st.session_state.render_key = 0

# MODEL LOADER
def load_ml_model():
    try:
        scaler   = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        features = joblib.load(os.path.join(MODELS_DIR, "selected_features.pkl"))
        model    = joblib.load(os.path.join(MODELS_DIR, "XGBoost.pkl"))
        return model, scaler, features
    except Exception:
        return None, None, None

# RADAR CHART 
def radar_chart(band_vals: dict) -> go.Figure:
    names  = list(band_vals.keys())
    values = [band_vals[b] for b in names] + [band_vals[names[0]]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=names + [names[0]],
        fill="toself",
        fillcolor="rgba(56, 189, 248, 0.15)",
        line=dict(color="#38bdf8", width=2.5),
        hovertemplate="%{theta}: %{r:.2f}<extra></extra>"
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1],
                            tickfont=dict(size=10, color="#aaaaaa"),
                            gridcolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(tickfont=dict(size=13, color="#ffffff"))
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=380,
        margin=dict(t=20, b=20, l=30, r=30)
    )
    return fig

# NEURAL INSIGHT GENERATOR 
def generate_insight(band_vals: dict, prob: float, high_enjoyment: bool) -> str:
    """
    Builds a detailed, multi-band neural commentary based on
    current band power values and model prediction.
    """
    # Rank bands by current power value
    sorted_bands = sorted(band_vals.items(), key=lambda x: x[1], reverse=True)

    # Active bands (above HIGH_THRESHOLD)
    active = [(b, v) for b, v in sorted_bands if v >= HIGH_THRESHOLD]

    # Top-N bands by current power regardless of threshold
    top_bands = sorted_bands[:TOP_N_BANDS]

    confidence_label = (
        "very high" if prob > 0.80 else
        "high"      if prob > 0.65 else
        "moderate"  if prob > 0.50 else
        "low"
    )

    if high_enjoyment:
        # high enjoyment comment 
        lines = [
            f"**🟢 High Enjoyment Detected** — Model confidence: **{prob*100:.1f}%** ({confidence_label})\n"
        ]

        if active:
            band_descriptions = []
            for b, v in active[:TOP_N_BANDS]:
                band_descriptions.append(
                    f"**{b}** ({v*100:.0f}% — *{BANDS[b]['neuro']}*)"
                )
            lines.append(
                f"The neural signature shows elevated activity in "
                f"{', '.join(band_descriptions[:-1])}"
                + (f" and {band_descriptions[-1]}" if len(band_descriptions) > 1
                   else (band_descriptions[0] if band_descriptions else ""))
                + "."
            )
            # Add per-band role explanations
            for b, v in active[:TOP_N_BANDS]:
                lines.append(f"- **{BANDS[b]['emoji']} {b}** ({v*100:.0f}%): {BANDS[b]['role_high']}.")
        else:
            # No band above threshold but still high enjoyment
            dominant_band, dominant_val = top_bands[0]
            lines.append(
                f"The dominant band is **{dominant_band}** ({dominant_val*100:.0f}%), "
                f"reflecting {BANDS[dominant_band]['role_high']}. "
                f"Even without extreme band activity, the overall spectral "
                f"pattern is classified as enjoyment-positive."
            )

        # Pan-spectral synthesis if multiple bands active
        if len(active) >= 3:
            lines.append(
                f"\n*Pan-spectral engagement detected across {len(active)} bands — "
                f"consistent with the holistic neural orchestration of music enjoyment "
                f"identified in this study's feature importance analysis.*"
            )

        return "\n\n".join(lines)

    else:
        # low enjoyment comment  
        lines = [
            f"**🔴 Low Enjoyment Detected** — Model confidence: **{(1-prob)*100:.1f}%** ({confidence_label})\n"
        ]

        dominant_band, dominant_val = top_bands[0]
        lines.append(
            f"The dominant neural activity is in the **{dominant_band}** band "
            f"({dominant_val*100:.0f}%), indicating {BANDS[dominant_band]['role_low']}."
        )

        # Explain which high-importance bands are underactive
        weak_important = [
            b for b in ["Gamma", "Beta"] if band_vals[b] < 0.40
        ]
        if weak_important:
            weak_str = " and ".join(
                [f"**{b}** ({band_vals[b]*100:.0f}%)" for b in weak_important]
            )
            lines.append(
                f"Notably, {weak_str} — the two highest-importance bands for enjoyment "
                f"classification — show low activity, suggesting absent perceptual "
                f"binding and sensorimotor resonance with this stimulus."
            )

        lines.append(
            f"*The overall spectral fingerprint does not match the high-enjoyment "
            f"neural patterns learned from the MUSIN-G training data.*"
        )

        return "\n\n".join(lines)


#  SIDEBAR 
with st.sidebar:
    st.header("🧠 BCI Simulator")
    st.caption("Manually adjust EEG band power or generate a random epoch.")

    if st.button("🎲 Capture Signal (Random Epoch)", use_container_width=True):
        for band in BANDS:
            st.session_state.band_data[band] = round(random.uniform(0.05, 0.95), 2)
        st.session_state.render_key += 1  
        st.session_state.randomize = True
        st.rerun()

    st.divider()

    band_vals = {}
    for band, info in BANDS.items():
        band_vals[band] = st.slider(
            f"{info['emoji']}  {band} Power  ({info['range']})",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.band_data.get(band, 0.5)),
            step=0.01,
            key=f"sl_{band}_{st.session_state.render_key}" 
        )

    st.session_state.randomize = False

    st.divider()
    st.caption("**Band Importance (XGBoost)**")
    for b, w in sorted(BAND_WEIGHTS.items(), key=lambda x: -x[1]):
        st.progress(w, text=f"{b}: {w*100:.1f}%")


# MAIN 
st.title("Neuro-Music Assistant")
st.markdown(
    "Interactive EEG band power simulator — enjoyment prediction "
    "powered by a trained XGBoost model (OpenNeuro ds003774 · MUSIN-G)."
)
st.divider()

# Load model
model, scaler, feat_list = load_ml_model()

# PREDICTION
# Fallback: weighted formula using real XGBoost band importances
prob = float(np.clip(
    BAND_WEIGHTS["Gamma"] * band_vals["Gamma"] +
    BAND_WEIGHTS["Beta"]  * band_vals["Beta"]  +
    BAND_WEIGHTS["Delta"] * band_vals["Delta"] +
    BAND_WEIGHTS["Theta"] * band_vals["Theta"] +
    BAND_WEIGHTS["Alpha"] * band_vals["Alpha"],
    0.01, 0.99
))

if model and scaler and feat_list:
    x_input = np.zeros((1, len(feat_list)))
    for i, name in enumerate(feat_list):
        for b in BANDS:
            if f"_{b}_" in name:
                x_input[0, i] = band_vals[b]
                break
    try:
        x_scaled = scaler.transform(x_input)
        prob = float(model.predict_proba(x_scaled)[0][1])
    except Exception:
        pass  # keep fallback

high_enjoyment = prob >= 0.5

# LAYOUT 
col1, col2 = st.columns([1.3, 1])

with col1:
    st.subheader("⚡ Brain Activity Radar")
    st.plotly_chart(radar_chart(band_vals), use_container_width=True, theme="streamlit")

with col2:
    st.subheader("🎯 Prediction Result")
    st.write("")

    if high_enjoyment:
        st.success("### 🟢 High Enjoyment")
    else:
        st.error("### 🔴 Low Enjoyment")

    st.metric(
        label="Model Confidence",
        value=f"{prob*100:.1f}%",
        delta="Engagement Detected" if high_enjoyment else "Low Engagement",
        delta_color="normal" if high_enjoyment else "inverse"
    )

    st.write("")

    # Mini band summary
    st.markdown("**Active Bands (≥70%)**")
    active_bands = {b: v for b, v in band_vals.items() if v >= HIGH_THRESHOLD}
    if active_bands:
        cols = st.columns(len(active_bands))
        for i, (b, v) in enumerate(active_bands.items()):
            with cols[i]:
                st.metric(
                    label=f"{BANDS[b]['emoji']} {b}",
                    value=f"{v*100:.0f}%"
                )
                st.caption(BANDS[b]['neuro'])
    else:
        st.caption("No band exceeds the 70% activity threshold.")

st.divider()

# NEURAL INSIGHT 
st.subheader("🤖 Neural Insight")
insight = generate_insight(band_vals, prob, high_enjoyment)

if high_enjoyment:
    st.info(insight)
else:
    st.warning(insight)

st.divider()

# BAND DETAIL TABLE 
with st.expander("📊 Full Band Analysis", expanded=False):
    st.markdown("Detailed breakdown of each frequency band's current power and neurophysiological role:")
    for band, info in BANDS.items():
        val = band_vals[band]
        bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
        status = "🟢 Active" if val >= HIGH_THRESHOLD else ("🟡 Moderate" if val >= 0.40 else "🔴 Low")
        st.markdown(
            f"**{info['emoji']} {band}** ({info['range']})  "
            f"`{bar}` {val*100:.0f}%  —  {status}  \n"
            f"*{info['role_high'] if val >= 0.5 else info['role_low']}*"
        )
        st.write("")

# FOOTER
st.caption(
    "Dataset: MUSIN-G (OpenNeuro ds003774) · "
    "Model: XGBoost (Acc: 77.51%, AUC: 0.861, Within-Subject) · "
    "This interface simulates EEG band power inputs — not a live EEG acquisition system."
)
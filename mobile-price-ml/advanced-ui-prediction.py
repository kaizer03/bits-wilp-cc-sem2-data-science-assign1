"""
Mobile Price Intelligence Dashboard
=====================================
Assignment 1 — Introduction to Data Science
M.Tech (Data Science & Engineering) · BITS Pilani WILP · Semester 2

Predicts mobile phone price range (0–3) using a trained Random Forest Classifier.
All 23 model features are preserved exactly. Rule-based analytics are clearly labeled.
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import streamlit.components.v1 as components

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mobile Price Intelligence",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
PRICE_LABELS = {
    0: ("Low Cost",          "#22d3ee", "💰"),
    1: ("Medium-Low Cost",   "#818cf8", "💵"),
    2: ("Medium-High Cost",  "#f472b6", "💳"),
    3: ("High Cost",         "#f59e0b", "💎"),
}

# Sensible defaults that give the UI an alive look from the start
DEFAULTS = dict(
    id_val=1, battery_power=1500, ram=2500, n_cores=4, clock_speed=2.0,
    px_height=800, px_width=1200, sc_h=12, sc_w=6, fc=8, pc=16,
    int_memory=64, talk_time=12, m_dep=0.5, mobile_wt=2, blue=1,
    dual_sim=1, four_g=1, three_g=1, touch_screen=1, wifi=1,
)

# Demo presets — selecting one populates all inputs via session state
PRESETS = {
    "Custom": None,
    "📦 Budget Phone": dict(
        id_val=1, battery_power=800, ram=512, n_cores=2, clock_speed=1.0,
        px_height=480, px_width=640, sc_h=9, sc_w=5, fc=2, pc=5,
        int_memory=8, talk_time=8, m_dep=0.8, mobile_wt=1, blue=0,
        dual_sim=0, four_g=0, three_g=1, touch_screen=1, wifi=1,
    ),
    "⚖️ Balanced Phone": dict(
        id_val=100, battery_power=1500, ram=2048, n_cores=4, clock_speed=2.0,
        px_height=900, px_width=1280, sc_h=12, sc_w=6, fc=8, pc=13,
        int_memory=32, talk_time=14, m_dep=0.5, mobile_wt=2, blue=1,
        dual_sim=1, four_g=1, three_g=1, touch_screen=1, wifi=1,
    ),
    "✨ Premium Phone": dict(
        id_val=500, battery_power=2500, ram=3500, n_cores=6, clock_speed=2.5,
        px_height=1920, px_width=1080, sc_h=14, sc_w=7, fc=13, pc=20,
        int_memory=64, talk_time=18, m_dep=0.4, mobile_wt=2, blue=1,
        dual_sim=1, four_g=1, three_g=1, touch_screen=1, wifi=1,
    ),
    "🚀 Flagship Phone": dict(
        id_val=999, battery_power=3500, ram=4000, n_cores=8, clock_speed=3.0,
        px_height=2160, px_width=1440, sc_h=16, sc_w=8, fc=20, pc=20,
        int_memory=128, talk_time=20, m_dep=0.3, mobile_wt=3, blue=1,
        dual_sim=1, four_g=1, three_g=1, touch_screen=1, wifi=1,
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE — initialize input keys with defaults (only on first load)
# ─────────────────────────────────────────────────────────────────────────────
for _k, _v in DEFAULTS.items():
    if f"inp_{_k}" not in st.session_state:
        st.session_state[f"inp_{_k}"] = _v
if "_last_preset" not in st.session_state:
    st.session_state["_last_preset"] = "Custom"

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — dark futuristic theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base & Background ── */
.stApp {
    background: linear-gradient(160deg, #050b14 0%, #07101d 55%, #0d1527 100%);
    background-attachment: fixed;
}
[data-testid="stAppViewContainer"] { background: transparent !important; }
.main .block-container {
    padding-top: 1rem;
    padding-bottom: 4rem;
    max-width: 1320px;
}
#MainMenu, footer { visibility: hidden; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #050b14; }
::-webkit-scrollbar-thumb { background: rgba(34,211,238,0.18); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(34,211,238,0.35); }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060d1a 0%, #0a1226 100%) !important;
    border-right: 1px solid rgba(34,211,238,0.09) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown { color: #94a3b8 !important; }

/* ── Primary Button ── */
.stButton > button {
    background: linear-gradient(135deg, #0e7490 0%, #1d4ed8 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 0.4px !important;
    padding: 0.55rem 1rem !important;
    box-shadow: 0 0 24px rgba(14,116,144,0.28), 0 2px 8px rgba(0,0,0,0.35) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 0 40px rgba(14,116,144,0.5), 0 4px 16px rgba(0,0,0,0.4) !important;
}
.stButton > button:active { transform: translateY(0px) !important; }

/* ── Number inputs ── */
.stNumberInput input {
    background: rgba(8,16,36,0.92) !important;
    border: 1px solid rgba(34,211,238,0.13) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    font-size: 0.9rem !important;
}
.stNumberInput input:focus {
    border-color: rgba(34,211,238,0.38) !important;
    box-shadow: 0 0 0 2px rgba(34,211,238,0.07) !important;
}

/* ── Select boxes ── */
.stSelectbox > div > div {
    background: rgba(8,16,36,0.92) !important;
    border: 1px solid rgba(34,211,238,0.13) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}

/* ── Widget labels ── */
.stNumberInput > label,
.stSelectbox > label {
    color: #4b5e7a !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.9px !important;
    text-transform: uppercase !important;
}

/* ── Progress bar fill ── */
[data-testid="stProgress"] > div {
    background: linear-gradient(90deg, #22d3ee, #818cf8) !important;
    border-radius: 8px !important;
}
.stProgress > div > div {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 8px !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 12px !important; }
[data-testid="stDataFrame"] iframe { border-radius: 12px !important; }

/* ────────────── Custom Layout Classes ────────────── */

/* Hero card */
.hero-card {
    background: linear-gradient(135deg, rgba(14,116,144,0.11) 0%, rgba(29,78,216,0.07) 100%);
    border: 1px solid rgba(34,211,238,0.2);
    border-radius: 24px;
    padding: 30px 36px 26px;
    margin-bottom: 1.4rem;
    box-shadow: 0 0 80px rgba(34,211,238,0.05), inset 0 1px 0 rgba(255,255,255,0.04);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
}
.hero-title {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #e2e8f0 30%, #22d3ee 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.03em;
    margin: 0 0 8px 0;
    line-height: 1.15;
}
.hero-sub {
    color: #64748b;
    font-size: 0.87rem;
    margin: 0 0 18px 0;
    line-height: 1.65;
    max-width: 820px;
}
.badge {
    display: inline-block;
    background: rgba(34,211,238,0.08);
    border: 1px solid rgba(34,211,238,0.22);
    color: #22d3ee;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 100px;
    margin-right: 7px;
}

/* Input section card */
.section-card {
    background: rgba(7,14,32,0.78);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 20px;
    padding: 20px 16px 16px;
    margin-bottom: 1rem;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    box-shadow: 0 4px 32px rgba(0,0,0,0.28);
}
.section-header {
    color: #64748b;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(34,211,238,0.09);
}

/* Derived metric cards */
.metric-card {
    background: rgba(7,14,32,0.9);
    border: 1px solid rgba(34,211,238,0.13);
    border-radius: 16px;
    padding: 18px 12px 14px;
    text-align: center;
    box-shadow: 0 0 28px rgba(34,211,238,0.04);
}
.metric-icon { font-size: 1.3rem; margin-bottom: 6px; }
.metric-label {
    color: #334155;
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 1.3px;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.metric-unit {
    color: #1e293b;
    font-size: 0.67rem;
    margin-top: 4px;
    font-weight: 500;
}

/* Prediction result card */
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 40px rgba(34,211,238,0.10), 0 0 0 1px rgba(34,211,238,0.28); }
    50%       { box-shadow: 0 0 72px rgba(34,211,238,0.24), 0 0 0 1px rgba(34,211,238,0.48); }
}
.pred-result-card {
    background: linear-gradient(135deg, rgba(14,116,144,0.17) 0%, rgba(99,102,241,0.11) 100%);
    border: 1px solid rgba(34,211,238,0.32);
    border-radius: 24px;
    padding: 36px 28px;
    text-align: center;
    animation: pulse-glow 3.2s ease-in-out infinite;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
}
.pred-icon { font-size: 3rem; margin-bottom: 12px; }
.pred-chip {
    display: inline-block;
    padding: 5px 18px;
    border-radius: 100px;
    font-size: 0.7rem;
    font-weight: 800;
    letter-spacing: 1.6px;
    text-transform: uppercase;
    margin-bottom: 14px;
    border: 1px solid;
    background: rgba(255,255,255,0.04);
}
.pred-label {
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 1;
    margin-bottom: 10px;
}
.pred-sub { color: #334155; font-size: 0.82rem; margin-top: 8px; }

/* Insight cards */
.insight-card {
    background: rgba(7,14,32,0.88);
    border-radius: 16px;
    padding: 15px 16px;
    border-left: 3px solid;
    box-shadow: 0 2px 14px rgba(0,0,0,0.22);
}
.insight-head {
    font-size: 0.67rem;
    font-weight: 800;
    letter-spacing: 1.3px;
    text-transform: uppercase;
    margin-bottom: 5px;
    opacity: 0.7;
}
.insight-val { font-size: 0.92rem; font-weight: 700; }
.insight-desc { font-size: 0.7rem; color: #334155; margin-top: 3px; }

/* Probability bars */
.prob-container { padding: 2px 0; }
.prob-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 11px;
    padding: 7px 10px;
    border-radius: 10px;
    transition: background 0.2s;
}
.prob-row.active-class {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.05);
}
.prob-class-label {
    width: 185px;
    font-size: 0.75rem;
    font-weight: 600;
    flex-shrink: 0;
    line-height: 1.3;
}
.prob-bar-bg {
    flex: 1;
    background: rgba(255,255,255,0.04);
    border-radius: 5px;
    height: 17px;
    overflow: hidden;
}
.prob-bar-fill { height: 100%; border-radius: 5px; }
.prob-pct {
    width: 50px;
    text-align: right;
    font-size: 0.82rem;
    font-weight: 800;
    flex-shrink: 0;
}

/* Interpretation box */
.interp-box {
    background: rgba(7,14,32,0.72);
    border: 1px solid rgba(99,102,241,0.16);
    border-radius: 16px;
    padding: 20px 24px;
    color: #94a3b8;
    font-size: 0.875rem;
    line-height: 1.85;
}

/* Section divider title */
.sec-title {
    color: #64748b;
    font-size: 0.71rem;
    font-weight: 800;
    letter-spacing: 1.6px;
    text-transform: uppercase;
    margin: 1.8rem 0 0.85rem 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.sec-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(34,211,238,0.22), transparent);
}

/* Spec score metadata */
.score-meta {
    color: #1e293b;
    font-size: 0.69rem;
    text-align: right;
    margin-top: 4px;
    font-style: italic;
}

/* Footer */
.footer-card {
    background: rgba(4,9,18,0.96);
    border: 1px solid rgba(99,102,241,0.07);
    border-radius: 14px;
    padding: 18px 28px;
    margin-top: 3.5rem;
    text-align: center;
    color: #1e293b;
    font-size: 0.77rem;
    line-height: 2.1;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
try:
    model = joblib.load("mobile_price_model.pkl")
except FileNotFoundError:
    st.error(
        "⚠️ Model file `mobile_price_model.pkl` not found. "
        "Ensure it is in the same directory as `advanced-ui-prediction.py` and re-run `streamlit run advanced-ui-prediction.py`."
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def compute_spec_score(ram, battery_power, screen_area, int_memory, fc, pc, clock_speed):
    """
    Rule-based composite device spec score in range 0–100.
    NOT a model output. Weights are fixed heuristics.
    """
    r = min(ram / 4096, 1.0)              * 0.30
    b = min(battery_power / 5000, 1.0)   * 0.18
    d = min(screen_area / 3_110_400, 1.0) * 0.22   # ref: 2160×1440
    m = min(int_memory / 256, 1.0)        * 0.15
    c = min((fc + pc) / 40, 1.0)          * 0.08
    s = min(clock_speed / 3.0, 1.0)       * 0.07
    return round((r + b + d + m + c + s) * 100, 1)


def tier_label(score):
    """Map spec score to a human-readable tier and accent color."""
    if score >= 75: return "Flagship Tier",  "#f59e0b"
    if score >= 50: return "Premium Tier",   "#f472b6"
    if score >= 28: return "Mid-Range Tier", "#818cf8"
    return "Budget Tier", "#22d3ee"


def get_display_strength(screen_area):
    if screen_area >= 2_000_000: return "Excellent", "#22d3ee"
    if screen_area >= 800_000:   return "Good",       "#818cf8"
    if screen_area >= 300_000:   return "Moderate",   "#f59e0b"
    return "Basic", "#64748b"


def get_memory_strength(ram, int_memory):
    if ram >= 3000 and int_memory >= 64: return "Powerful", "#22d3ee"
    if ram >= 1500 and int_memory >= 16: return "Adequate", "#818cf8"
    return "Limited", "#64748b"


def get_connectivity(four_g, three_g, wifi, blue, dual_sim):
    score = four_g + three_g + wifi + blue + dual_sim
    if score >= 5: return "Full-Featured", "#22d3ee"
    if score >= 3: return "Standard",      "#818cf8"
    if score >= 1: return "Basic",         "#f59e0b"
    return "Minimal", "#64748b"


def get_power_efficiency(battery_per_core, talk_time):
    if battery_per_core >= 500 and talk_time >= 16: return "Excellent", "#22d3ee"
    if battery_per_core >= 250 and talk_time >= 10: return "Good",      "#818cf8"
    if battery_per_core >= 100:                     return "Fair",       "#f59e0b"
    return "Weak", "#64748b"


def build_interpretation(ram, battery_power, screen_area, int_memory, four_g, three_g, fc, pc):
    """
    Assemble a plain-language interpretation from input values using fixed rules.
    This is NOT derived from model internals, SHAP, or feature importance.
    """
    parts = []
    # RAM insight
    if ram >= 3000:
        parts.append("High RAM capacity signals strong multitasking performance and premium market positioning.")
    elif ram >= 1500:
        parts.append("Moderate RAM suggests a capable mid-range experience suitable for everyday use.")
    else:
        parts.append("Limited RAM is characteristic of entry-level or budget-oriented devices.")
    # Display insight
    if screen_area >= 1_500_000:
        parts.append("A large, high-resolution display adds considerable perceived value to the device.")
    elif screen_area >= 500_000:
        parts.append("Moderate display resolution is consistent with the expected price segment.")
    else:
        parts.append("A compact or low-resolution display typically constrains the upper price ceiling.")
    # Connectivity insight
    if four_g:
        parts.append("4G LTE connectivity reinforces a modern, well-connected feature set.")
    elif three_g:
        parts.append("3G-only connectivity places the device closer to an older or lower-cost segment.")
    else:
        parts.append("Lack of mobile data capability is uncommon for most current price tiers.")
    # Storage insight
    if int_memory >= 64:
        parts.append("Generous internal storage is aligned with mid-to-premium pricing expectations.")
    elif int_memory >= 16:
        parts.append("Standard storage capacity fits comfortably within the mid-range segment.")
    else:
        parts.append("Minimal onboard storage is a common trait of budget smartphones.")
    # Camera insight
    if (fc + pc) >= 30:
        parts.append("Strong combined camera specs further support a higher predicted price class.")
    elif (fc + pc) >= 15:
        parts.append("Capable dual-camera configuration is consistent with the predicted tier.")
    return " ".join(parts)


def render_prob_bars(probabilities, predicted_class):
    """Return custom-styled HTML for class probability bars."""
    html = '<div class="prob-container">'
    for cls_idx in range(4):
        label_text, color, icon = PRICE_LABELS[cls_idx]
        pct = probabilities[cls_idx] * 100
        active_cls = "active-class" if cls_idx == predicted_class else ""
        html += f"""
        <div class="prob-row {active_cls}">
            <div class="prob-class-label" style="color:{color};">
                {icon} Class {cls_idx} &nbsp;·&nbsp; {label_text}
            </div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill"
                     style="width:{pct:.1f}%;
                            background:linear-gradient(90deg,{color}55,{color}cc);">
                </div>
            </div>
            <div class="prob-pct" style="color:{color};">{pct:.1f}%</div>
        </div>"""
    html += "</div>"
    return html


def metric_card(icon, label, value, unit="", color="#22d3ee"):
    return f"""
    <div class="metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color};">{value}</div>
        <div class="metric-unit">{unit}</div>
    </div>"""


def insight_card(icon, title, value, desc, color):
    return f"""
    <div class="insight-card" style="border-left-color:{color};">
        <div class="insight-head" style="color:{color};">{icon}&nbsp; {title}</div>
        <div class="insight-val" style="color:{color};">{value}</div>
        <div class="insight-desc">{desc}</div>
    </div>"""


# ── Widget callbacks ──────────────────────────────────────────────────────────
# Callbacks fire BEFORE the next script rerun, so session state writes are safe
# even though the widget keys will be instantiated moments later.

def _apply_preset():
    """Called when the preset selectbox changes."""
    preset_name = st.session_state["preset_selector"]
    if preset_name != "Custom" and PRESETS.get(preset_name):
        for k, v in PRESETS[preset_name].items():
            st.session_state[f"inp_{k}"] = v
    st.session_state["_last_preset"] = preset_name


def _reset_to_defaults():
    """Called when the Reset button is clicked."""
    for k, v in DEFAULTS.items():
        st.session_state[f"inp_{k}"] = v
    st.session_state["preset_selector"] = "Custom"
    st.session_state["_last_preset"] = "Custom"


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:14px 0 6px;">
        <div style="font-size:1.2rem;font-weight:800;color:#e2e8f0;letter-spacing:-0.02em;">
            📱 MPI Dashboard
        </div>
        <div style="font-size:0.7rem;color:#334155;letter-spacing:0.5px;margin-top:3px;">
            Mobile Price Intelligence
        </div>
    </div>
    <hr style="border:none;border-top:1px solid rgba(34,211,238,0.09);margin:10px 0 14px;">
    <div style="font-size:0.68rem;font-weight:800;letter-spacing:1.3px;color:#334155;
                text-transform:uppercase;margin-bottom:10px;">Model Summary</div>
    <div style="background:rgba(8,16,36,0.82);border:1px solid rgba(34,211,238,0.09);
                border-radius:12px;padding:12px 14px;margin-bottom:18px;">
        <div style="font-size:0.79rem;color:#64748b;line-height:2;">
            <b style="color:#22d3ee;">Algorithm</b><br>Random Forest Classifier<br>
            <b style="color:#22d3ee;">Output Classes</b><br>4 Price Ranges (0 – 3)<br>
            <b style="color:#22d3ee;">Input Features</b><br>23 (21 raw + 2 derived)<br>
            <b style="color:#22d3ee;">Derived Features</b><br>
            <code style="font-size:0.72rem;">screen_area</code>,
            <code style="font-size:0.72rem;">battery_per_core</code>
        </div>
    </div>
    <div style="font-size:0.68rem;font-weight:800;letter-spacing:1.3px;color:#334155;
                text-transform:uppercase;margin-bottom:10px;">Price Range Legend</div>
    """, unsafe_allow_html=True)

    for cls, (label, color, icon) in PRICE_LABELS.items():
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;padding:7px 10px;
                    margin-bottom:6px;background:rgba(8,16,36,0.72);
                    border-left:3px solid {color};border-radius:8px;">
            <span style="font-size:1rem;">{icon}</span>
            <div>
                <div style="font-size:0.69rem;font-weight:800;color:{color};letter-spacing:0.4px;">
                    Class {cls}</div>
                <div style="font-size:0.74rem;color:#334155;">{label}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("ℹ️ How Prediction Works"):
        st.markdown("""
        <div style="font-size:0.79rem;color:#475569;line-height:1.75;">

        <b style="color:#94a3b8;">Model Output</b><br>
        The price class (0–3) and class probabilities come directly from
        <code>model.predict()</code> and <code>model.predict_proba()</code>.
        These are genuine model outputs.<br><br>

        <b style="color:#94a3b8;">Derived Features</b><br>
        <code>screen_area = px_height × px_width</code><br>
        <code>battery_per_core = battery_power / n_cores</code><br>
        These are computed automatically and included in the model input.<br><br>

        <b style="color:#94a3b8;">Rule-Based Analytics</b><br>
        Spec Score, insight cards, and interpretation text are computed
        from input values using fixed thresholds. They are <i>not</i>
        model outputs and do not affect the prediction.
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HERO SECTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-card">
    <div class="hero-title">Mobile Price Intelligence</div>
    <p class="hero-sub">
        Predict a mobile phone's price range from its technical specifications using a trained
        Random Forest Classifier. Configure device specs below, then run the prediction engine
        for a full analytics breakdown — including class probabilities, spec strength, and
        derived device insights.
    </p>
    <span class="badge">Random Forest</span>
    <span class="badge">23 Features</span>
    <span class="badge">4 Price Classes</span>
    <span class="badge">predict_proba</span>
    <span class="badge">M.Tech BITS Pilani WILP</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DEMO PRESET SELECTOR
# ─────────────────────────────────────────────────────────────────────────────
preset_col, _ = st.columns([2, 5])
with preset_col:
    st.selectbox(
        "Demo Preset",
        options=list(PRESETS.keys()),
        key="preset_selector",
        on_change=_apply_preset,
        help="Load a predefined device profile to auto-populate all inputs.",
    )

st.markdown(
    "<hr style='border:none;border-top:1px solid rgba(34,211,238,0.07);margin:0.8rem 0 1rem;'>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# INPUT SECTIONS — three main spec cards
# ─────────────────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">⚙️ Core Specifications</div>', unsafe_allow_html=True)
    id_val        = st.number_input("Mobile ID",           min_value=1,   step=1,            key="inp_id_val")
    battery_power = st.number_input("Battery Power (mAh)", min_value=0,   step=1,            key="inp_battery_power")
    ram           = st.number_input("RAM (MB)",            min_value=0,   step=1,            key="inp_ram")
    n_cores       = st.number_input("Number of Cores",     min_value=1,   step=1,            key="inp_n_cores")
    clock_speed   = st.number_input("Clock Speed (GHz)",   min_value=0.0, step=0.1,
                                    format="%.1f",                                           key="inp_clock_speed")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🖥️ Display & Camera</div>', unsafe_allow_html=True)
    px_height = st.number_input("Pixel Height (px)",    min_value=0, step=1, key="inp_px_height")
    px_width  = st.number_input("Pixel Width (px)",     min_value=0, step=1, key="inp_px_width")
    sc_h      = st.number_input("Screen Height (cm)",   min_value=0, step=1, key="inp_sc_h")
    sc_w      = st.number_input("Screen Width (cm)",    min_value=0, step=1, key="inp_sc_w")
    fc        = st.number_input("Front Camera (MP)",    min_value=0, step=1, key="inp_fc")
    pc        = st.number_input("Primary Camera (MP)",  min_value=0, step=1, key="inp_pc")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📡 Connectivity & Build</div>', unsafe_allow_html=True)
    int_memory = st.number_input("Internal Memory (GB)", min_value=0,   step=1,   key="inp_int_memory")
    talk_time  = st.number_input("Talk Time (hrs)",      min_value=0,   step=1,   key="inp_talk_time")
    m_dep      = st.number_input("Depth (cm)",           min_value=0.0, step=0.1,
                                 format="%.1f",                                   key="inp_m_dep")
    mobile_wt  = st.selectbox(
        "Weight Category",
        options=[1, 2, 3],
        format_func=lambda x: {1: "Light", 2: "Medium", 3: "Heavy"}[x],
        key="inp_mobile_wt",
    )
    blue     = st.selectbox("Bluetooth", options=[0, 1],
                            format_func=lambda x: "✓ Yes" if x else "✗ No", key="inp_blue")
    dual_sim = st.selectbox("Dual SIM",  options=[0, 1],
                            format_func=lambda x: "✓ Yes" if x else "✗ No", key="inp_dual_sim")
    st.markdown('</div>', unsafe_allow_html=True)

# Network & Features row
st.markdown('<div class="section-card" style="padding:14px 16px 12px;">', unsafe_allow_html=True)
st.markdown('<div class="section-header">🔗 Network & Features</div>', unsafe_allow_html=True)
nc1, nc2, nc3, nc4 = st.columns(4)
with nc1:
    four_g       = st.selectbox("4G LTE",       options=[0, 1],
                                format_func=lambda x: "✓ Yes" if x else "✗ No", key="inp_four_g")
with nc2:
    three_g      = st.selectbox("3G",           options=[0, 1],
                                format_func=lambda x: "✓ Yes" if x else "✗ No", key="inp_three_g")
with nc3:
    touch_screen = st.selectbox("Touch Screen", options=[0, 1],
                                format_func=lambda x: "✓ Yes" if x else "✗ No", key="inp_touch_screen")
with nc4:
    wifi         = st.selectbox("WiFi",         options=[0, 1],
                                format_func=lambda x: "✓ Yes" if x else "✗ No", key="inp_wifi")
st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DERIVED FEATURES  (computed from inputs, passed to model as features 22 & 23)
# ─────────────────────────────────────────────────────────────────────────────
screen_area      = px_height * px_width
battery_per_core = battery_power / n_cores if n_cores != 0 else 0
camera_total     = fc + pc

# ─────────────────────────────────────────────────────────────────────────────
# DERIVED ANALYTICS STRIP
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-title">📊 Derived Feature Analytics</div>', unsafe_allow_html=True)
ma1, ma2, ma3, ma4 = st.columns(4)
with ma1:
    st.markdown(metric_card("📐", "Screen Area",     f"{screen_area:,}",           "px²",      "#22d3ee"), unsafe_allow_html=True)
with ma2:
    st.markdown(metric_card("⚡", "Battery / Core",  f"{battery_per_core:.0f}",    "mAh/core", "#818cf8"), unsafe_allow_html=True)
with ma3:
    st.markdown(metric_card("🧠", "RAM",             f"{ram:,}",                   "MB",       "#f472b6"), unsafe_allow_html=True)
with ma4:
    st.markdown(metric_card("📷", "Combined Camera", f"{camera_total}",            "MP total", "#f59e0b"), unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ACTION BUTTONS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
btn1, btn2, btn3 = st.columns([3, 1, 3])
with btn1:
    predict_clicked = st.button("🚀 Run Prediction Engine", use_container_width=True)
with btn2:
    st.button("↺ Reset", use_container_width=True, on_click=_reset_to_defaults)

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION SECTION  (rendered only after clicking Predict)
# ─────────────────────────────────────────────────────────────────────────────
if predict_clicked:

    # Scroll anchor + best-effort JS scroll to result
    st.markdown('<div id="prediction-result"></div>', unsafe_allow_html=True)
    components.html("""
    <script>
        setTimeout(function() {
            try {
                var el = window.parent.document.getElementById('prediction-result');
                if (el) { el.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
            } catch(e) {}
        }, 350);
    </script>
    """, height=0)

    # ── Build exact 23-feature input DataFrame (column order matches training) ──
    input_df = pd.DataFrame([{
        "id":               id_val,
        "battery_power":    battery_power,
        "blue":             blue,
        "clock_speed":      clock_speed,
        "dual_sim":         dual_sim,
        "fc":               fc,
        "four_g":           four_g,
        "int_memory":       int_memory,
        "m_dep":            m_dep,
        "mobile_wt":        mobile_wt,
        "n_cores":          n_cores,
        "pc":               pc,
        "px_height":        px_height,
        "px_width":         px_width,
        "ram":              ram,
        "sc_h":             sc_h,
        "sc_w":             sc_w,
        "talk_time":        talk_time,
        "three_g":          three_g,
        "touch_screen":     touch_screen,
        "wifi":             wifi,
        "screen_area":      screen_area,
        "battery_per_core": battery_per_core,
    }])

    # ── Model inference ──────────────────────────────────────────────────────
    prediction        = int(model.predict(input_df)[0])
    pred_label, pred_color, pred_icon = PRICE_LABELS[prediction]

    has_proba     = hasattr(model, "predict_proba")
    probabilities = model.predict_proba(input_df)[0] if has_proba else None

    # ── Rule-based analytics ─────────────────────────────────────────────────
    spec_score          = compute_spec_score(ram, battery_power, screen_area, int_memory, fc, pc, clock_speed)
    tier_name, tier_clr = tier_label(spec_score)

    disp_val,  disp_clr  = get_display_strength(screen_area)
    mem_val,   mem_clr   = get_memory_strength(ram, int_memory)
    conn_val,  conn_clr  = get_connectivity(four_g, three_g, wifi, blue, dual_sim)
    power_val, power_clr = get_power_efficiency(battery_per_core, talk_time)
    interpretation       = build_interpretation(ram, battery_power, screen_area, int_memory,
                                                four_g, three_g, fc, pc)

    st.markdown(
        "<hr style='border:none;border-top:1px solid rgba(34,211,238,0.07);margin:1.5rem 0;'>",
        unsafe_allow_html=True,
    )

    # ── Input Snapshot ───────────────────────────────────────────────────────
    st.markdown('<div class="sec-title">📋 Input Snapshot</div>', unsafe_allow_html=True)
    st.dataframe(input_df, use_container_width=True)

    # ── Spec Strength Score ──────────────────────────────────────────────────
    st.markdown('<div class="sec-title">📈 Device Spec Strength</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px;">
        <span style="font-size:0.8rem;color:{tier_clr};font-weight:700;">{tier_name}</span>
        <span style="font-size:1.35rem;color:{tier_clr};font-weight:800;
                     letter-spacing:-0.02em;">{spec_score} <span style="font-size:0.75rem;
                     color:#334155;font-weight:500;">/ 100</span></span>
    </div>
    """, unsafe_allow_html=True)
    st.progress(spec_score / 100)
    st.markdown(
        '<div class="score-meta">'
        '★ Rule-based composite — not a model output. '
        'Weights: RAM 30 · Display 22 · Battery 18 · Storage 15 · Camera 8 · Clock 7.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Prediction Result Card ───────────────────────────────────────────────
    st.markdown('<div class="sec-title">🎯 Prediction Result</div>', unsafe_allow_html=True)
    top_prob_str = (
        f"Top class probability: {probabilities[prediction]*100:.1f}%"
        if has_proba else ""
    )
    st.markdown(f"""
    <div class="pred-result-card">
        <div class="pred-icon">{pred_icon}</div>
        <div class="pred-chip"
             style="color:{pred_color};border-color:{pred_color};">
            Price Class {prediction}
        </div>
        <div class="pred-label" style="color:{pred_color};">{pred_label}</div>
        <div class="pred-sub">
            Predicted by Random Forest Classifier
            {'&nbsp;·&nbsp;' + top_prob_str if top_prob_str else ''}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Class Probability Analysis ───────────────────────────────────────────
    if has_proba:
        st.markdown('<div class="sec-title">📊 Class Probability Analysis</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.72rem;color:#1e293b;margin-bottom:10px;">'
            'Source: <code>model.predict_proba()</code> — genuine model output, not estimated.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(render_prob_bars(probabilities, prediction), unsafe_allow_html=True)

    # ── Derived Insight Cards ────────────────────────────────────────────────
    st.markdown('<div class="sec-title">💡 Device Insight Cards</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.7rem;color:#1e293b;margin-bottom:12px;">'
        '★ Rule-based — computed from your input values, not from the model.</div>',
        unsafe_allow_html=True,
    )
    ic1, ic2, ic3, ic4 = st.columns(4)
    with ic1:
        st.markdown(insight_card(
            "🖥️", "Display Strength", disp_val,
            f"Screen area: {screen_area:,} px²",
            disp_clr,
        ), unsafe_allow_html=True)
    with ic2:
        st.markdown(insight_card(
            "🧠", "Memory Strength", mem_val,
            f"RAM: {ram:,} MB · Storage: {int_memory} GB",
            mem_clr,
        ), unsafe_allow_html=True)
    with ic3:
        st.markdown(insight_card(
            "📡", "Connectivity", conn_val,
            f"4G:{four_g} · 3G:{three_g} · WiFi:{wifi} · BT:{blue} · DS:{dual_sim}",
            conn_clr,
        ), unsafe_allow_html=True)
    with ic4:
        st.markdown(insight_card(
            "⚡", "Power Efficiency", power_val,
            f"{battery_per_core:.0f} mAh/core · {talk_time} h talk time",
            power_clr,
        ), unsafe_allow_html=True)

    # ── Auto-Generated Interpretation ───────────────────────────────────────
    st.markdown('<div class="sec-title">💬 Auto-Generated Interpretation</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="interp-box">'
        f'<span style="font-size:0.68rem;color:#1e293b;font-weight:800;letter-spacing:1px;'
        f'text-transform:uppercase;">Rule-based · Not model-derived</span><br><br>'
        f'{interpretation}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── How-It-Works Expander ────────────────────────────────────────────────
    with st.expander("📖 How This Prediction Is Interpreted"):
        st.markdown("""
        <div style="font-size:0.83rem;color:#475569;line-height:1.8;">

        <b style="color:#94a3b8;">Prediction (Model Output)</b><br>
        The predicted class (0–3) and class probabilities come directly from the
        trained <b>Random Forest Classifier</b> via <code>model.predict()</code> and
        <code>model.predict_proba()</code>. These are genuine model outputs.<br><br>

        <b style="color:#94a3b8;">Spec Strength Score (Rule-Based)</b><br>
        The score is a weighted linear combination of normalized input values.
        It is intended as a human-readable device tier indicator and does
        <em>not</em> influence or correlate directly with the model's prediction.<br><br>

        <b style="color:#94a3b8;">Insight Cards (Rule-Based)</b><br>
        Display Strength, Memory Strength, Connectivity, and Power Efficiency
        are computed using fixed thresholds on your input values.
        They provide supplementary context only.<br><br>

        <b style="color:#94a3b8;">Derived Features (Model Input)</b><br>
        <code>screen_area = px_height × px_width</code><br>
        <code>battery_per_core = battery_power / n_cores</code><br>
        These two derived features are included as features 22 and 23 in the
        model input DataFrame alongside the 21 raw inputs.

        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-card">
    <b style="color:#334155;">Introduction to Data Science — Assignment 1</b><br>
    M.Tech (Data Science &amp; Engineering) &nbsp;·&nbsp; BITS Pilani WILP &nbsp;·&nbsp; Semester 2<br>
    Model: Random Forest Classifier &nbsp;·&nbsp; Dataset: Mobile Price Classification<br>
    <span style="font-size:0.7rem;color:#0f172a;">
        Built with Streamlit &nbsp;·&nbsp; Powered by scikit-learn &nbsp;·&nbsp; For academic use only
    </span>
</div>
""", unsafe_allow_html=True)

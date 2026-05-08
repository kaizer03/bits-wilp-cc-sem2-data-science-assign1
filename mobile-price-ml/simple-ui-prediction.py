import streamlit as st
import joblib
import numpy as np
import pandas as pd
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Mobile Price Prediction",
    page_icon="📱",
    layout="wide"
)

# ---------- Presets (identical names & specs to advanced-ui-prediction.py) ----------
PRESETS = {
    "Custom": None,
    "📦 Budget Basic": dict(
        battery_power=700,  ram=300,  n_cores=2, clock_speed=0.8,
        px_height=400,  px_width=600,  sc_h=8,  sc_w=4,  fc=1,  pc=3,
        int_memory=4,   talk_time=6,  m_dep=0.9, mobile_wt=1,
        blue=0, dual_sim=0, four_g=0, three_g=0, touch_screen=1, wifi=0,
    ),
    "📦 Budget Phone": dict(
        battery_power=800,  ram=512,  n_cores=2, clock_speed=1.0,
        px_height=480,  px_width=640,  sc_h=9,  sc_w=5,  fc=2,  pc=5,
        int_memory=8,   talk_time=8,  m_dep=0.8, mobile_wt=1,
        blue=0, dual_sim=0, four_g=0, three_g=1, touch_screen=1, wifi=1,
    ),
    "⚖️ Entry Mid": dict(
        battery_power=1100, ram=1000, n_cores=4, clock_speed=1.5,
        px_height=720,  px_width=900,  sc_h=10, sc_w=5,  fc=5,  pc=8,
        int_memory=16,  talk_time=10, m_dep=0.7, mobile_wt=2,
        blue=1, dual_sim=1, four_g=0, three_g=1, touch_screen=1, wifi=1,
    ),
    "⚖️ Balanced Phone": dict(
        battery_power=1500, ram=2048, n_cores=4, clock_speed=2.0,
        px_height=900,  px_width=1280, sc_h=12, sc_w=6,  fc=8,  pc=13,
        int_memory=32,  talk_time=14, m_dep=0.5, mobile_wt=2,
        blue=1, dual_sim=1, four_g=1, three_g=1, touch_screen=1, wifi=1,
    ),
    "⚖️ Mid Plus": dict(
        battery_power=1700, ram=1800, n_cores=4, clock_speed=2.0,
        px_height=900,  px_width=1280, sc_h=11, sc_w=6,  fc=8,  pc=12,
        int_memory=32,  talk_time=14, m_dep=0.5, mobile_wt=2,
        blue=1, dual_sim=1, four_g=1, three_g=1, touch_screen=1, wifi=1,
    ),
    "✨ Upper Mid": dict(
        battery_power=1900, ram=2600, n_cores=6, clock_speed=2.2,
        px_height=960,  px_width=1600, sc_h=13, sc_w=7,  fc=10, pc=16,
        int_memory=64,  talk_time=16, m_dep=0.5, mobile_wt=2,
        blue=1, dual_sim=1, four_g=1, three_g=1, touch_screen=1, wifi=1,
    ),
    "✨ Premium Phone": dict(
        battery_power=2500, ram=3500, n_cores=6, clock_speed=2.5,
        px_height=1920, px_width=1080, sc_h=14, sc_w=7,  fc=13, pc=20,
        int_memory=64,  talk_time=18, m_dep=0.4, mobile_wt=2,
        blue=1, dual_sim=1, four_g=1, three_g=1, touch_screen=1, wifi=1,
    ),
    "✨ Premium Pro": dict(
        battery_power=1998, ram=3800, n_cores=8, clock_speed=2.7,
        px_height=1600, px_width=1800, sc_h=15, sc_w=7,  fc=16, pc=20,
        int_memory=64,  talk_time=19, m_dep=0.4, mobile_wt=2,
        blue=1, dual_sim=1, four_g=1, three_g=1, touch_screen=1, wifi=1,
    ),
    "🚀 Flagship Phone": dict(
        battery_power=3500, ram=4000, n_cores=8, clock_speed=3.0,
        px_height=2160, px_width=1440, sc_h=16, sc_w=8,  fc=20, pc=20,
        int_memory=128, talk_time=20, m_dep=0.3, mobile_wt=3,
        blue=1, dual_sim=1, four_g=1, three_g=1, touch_screen=1, wifi=1,
    ),
    "🚀 Ultra Flagship": dict(
        battery_power=1998, ram=3998, n_cores=8, clock_speed=3.0,
        px_height=1960, px_width=1998, sc_h=17, sc_w=8,  fc=19, pc=20,
        int_memory=64,  talk_time=20, m_dep=0.3, mobile_wt=3,
        blue=1, dual_sim=1, four_g=1, three_g=1, touch_screen=1, wifi=1,
    ),
}

_DEFAULTS = PRESETS["⚖️ Balanced Phone"]
for _k, _v in _DEFAULTS.items():
    if f"inp_{_k}" not in st.session_state:
        st.session_state[f"inp_{_k}"] = _v
if "_last_preset" not in st.session_state:
    st.session_state["_last_preset"] = "Custom"


def _apply_preset():
    name = st.session_state["preset_sel"]
    if name != "Custom" and PRESETS.get(name):
        for k, v in PRESETS[name].items():
            st.session_state[f"inp_{k}"] = v
    st.session_state["_last_preset"] = name


# ---------- Custom CSS ----------
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #07111f 0%, #0b1728 45%, #111827 100%);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    .hero-card {
        background: rgba(17, 24, 39, 0.78);
        border: 1px solid rgba(0, 255, 255, 0.18);
        border-radius: 20px;
        padding: 24px 28px;
        box-shadow: 0 0 30px rgba(0, 255, 255, 0.08);
        margin-bottom: 1.5rem;
    }
    .section-card {
        background: rgba(17, 24, 39, 0.72);
        border: 1px solid rgba(99, 102, 241, 0.22);
        border-radius: 18px;
        padding: 20px;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(34, 211, 238, 0.22);
        border-radius: 16px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 0 20px rgba(34, 211, 238, 0.06);
    }
    .pred-box {
        background: linear-gradient(135deg, rgba(16,185,129,0.18), rgba(34,211,238,0.12));
        border: 1px solid rgba(16,185,129,0.35);
        border-radius: 18px;
        padding: 18px;
        margin-top: 1rem;
        font-size: 1.2rem;
        font-weight: 600;
    }
    h1, h2, h3 {
        letter-spacing: 0.2px;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Load model ----------
import train as _train

try:
    if not __import__("os").path.exists(_train.MODEL_PATH):
        with st.spinner("Training model for the first time — this takes ~10 seconds…"):
            _train.train_and_save()
    model = joblib.load(_train.MODEL_PATH)
except Exception as _e:
    st.error(f"⚠️ Could not load or train the model: {_e}")
    st.stop()

# ---------- Header ----------
st.markdown("""
<div class="hero-card">
    <h1>📱 Mobile Price Prediction</h1>
    <p style="font-size:1.05rem; color:#cbd5e1; margin-bottom:0;">
        Predict the mobile phone price range using technical specifications.
        This Streamlit app uses the trained <b>Random Forest Classifier</b> built in the assignment notebook.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
st.sidebar.title("About")
st.sidebar.info(
    "Enter the mobile specifications, click predict, and the app will return the predicted "
    "price range: 0, 1, 2, or 3."
)
st.sidebar.markdown("### Price Range Meaning")
st.sidebar.write("**0** → Low")
st.sidebar.write("**1** → Medium-Low")
st.sidebar.write("**2** → Medium-High")
st.sidebar.write("**3** → High")

# ---------- Phone Profile Selector ----------
_pc, _ = st.columns([2, 5])
with _pc:
    st.selectbox(
        "Phone Profile",
        options=list(PRESETS.keys()),
        key="preset_sel",
        on_change=_apply_preset,
        help="Load a sample phone profile — inputs will be populated automatically.",
    )
st.markdown(
    "<hr style='border:none;border-top:1px solid rgba(0,255,255,0.08);margin:0.2rem 0 1rem;'>",
    unsafe_allow_html=True,
)

# ---------- Input layout ----------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Core Specs")
    battery_power = st.number_input("Battery Power (mAh)", min_value=0, step=1,            key="inp_battery_power")
    ram           = st.number_input("RAM (MB)",             min_value=0, step=1,            key="inp_ram")
    n_cores       = st.number_input("Number of Cores",      min_value=1, step=1,            key="inp_n_cores")
    clock_speed   = st.number_input("Clock Speed (GHz)",    min_value=0.0, step=0.1,
                                    format="%.1f",                                          key="inp_clock_speed")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Display & Camera")
    px_height = st.number_input("Pixel Height (px)",   min_value=0, step=1, key="inp_px_height")
    px_width  = st.number_input("Pixel Width (px)",    min_value=0, step=1, key="inp_px_width")
    sc_h      = st.number_input("Screen Height (cm)",  min_value=0, step=1, key="inp_sc_h")
    sc_w      = st.number_input("Screen Width (cm)",   min_value=0, step=1, key="inp_sc_w")
    fc        = st.number_input("Front Camera (MP)",   min_value=0, step=1, key="inp_fc")
    pc        = st.number_input("Primary Camera (MP)", min_value=0, step=1, key="inp_pc")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Connectivity & Storage")
    int_memory = st.number_input("Internal Memory (GB)", min_value=0,   step=1,            key="inp_int_memory")
    talk_time  = st.number_input("Talk Time (hrs)",      min_value=0,   step=1,            key="inp_talk_time")
    m_dep      = st.number_input("Mobile Depth (cm)",    min_value=0.0, step=0.1,
                                 format="%.1f",                                            key="inp_m_dep")
    mobile_wt  = st.selectbox("Mobile Weight Category", options=[1, 2, 3],
                               format_func=lambda x: {1:"Low", 2:"Med", 3:"High"}[x],    key="inp_mobile_wt")
    blue       = st.selectbox("Bluetooth", options=[0, 1],
                               format_func=lambda x: "Yes" if x == 1 else "No",           key="inp_blue")
    dual_sim   = st.selectbox("Dual SIM",  options=[0, 1],
                               format_func=lambda x: "Yes" if x == 1 else "No",           key="inp_dual_sim")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Additional Features")

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    four_g       = st.selectbox("4G",           options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key="inp_four_g")
with c2:
    three_g      = st.selectbox("3G",           options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key="inp_three_g")
with c3:
    touch_screen = st.selectbox("Touch Screen", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key="inp_touch_screen")
with c4:
    wifi         = st.selectbox("WiFi",         options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key="inp_wifi")
with c5:
    st.markdown("")
    st.markdown("")
    predict_clicked = st.button("🚀 Predict Price Range", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Derived features ----------
screen_area = px_height * px_width
battery_per_core = battery_power / n_cores if n_cores != 0 else 0

# ---------- Metrics preview ----------
m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(f'<div class="metric-card"><h3>Screen Area</h3><p>{screen_area}</p></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-card"><h3>Battery / Core</h3><p>{battery_per_core:.2f}</p></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric-card"><h3>RAM</h3><p>{ram}</p></div>', unsafe_allow_html=True)

# ---------- Prediction ----------
if predict_clicked:
    # Scroll anchor — JS attempts to scroll the parent window to this element
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

    input_df = pd.DataFrame([{
        "id": 1,
        "battery_power": battery_power,
        "blue": blue,
        "clock_speed": clock_speed,
        "dual_sim": dual_sim,
        "fc": fc,
        "four_g": four_g,
        "int_memory": int_memory,
        "m_dep": m_dep,
        "mobile_wt": mobile_wt,
        "n_cores": n_cores,
        "pc": pc,
        "px_height": px_height,
        "px_width": px_width,
        "ram": ram,
        "sc_h": sc_h,
        "sc_w": sc_w,
        "talk_time": talk_time,
        "three_g": three_g,
        "touch_screen": touch_screen,
        "wifi": wifi,
        "screen_area": screen_area,
        "battery_per_core": battery_per_core
    }])

    prediction = int(model.predict(input_df)[0])

    label_map = {
        0: ("Low Cost", "💚", "~₹5,000 – ₹10,000", "#10b981"),
        1: ("Medium-Low Cost", "🔵", "~₹10,000 – ₹20,000", "#3b82f6"),
        2: ("Medium-High Cost", "🟠", "~₹20,000 – ₹35,000", "#f97316"),
        3: ("High Cost", "🔴", "~₹35,000+", "#ef4444"),
    }

    # ---- Price range visual tier display ----
    st.markdown("### 🎯 Prediction Result")
    tier_cols = st.columns(4)
    for tier, col in enumerate(tier_cols):
        name, icon, price_range, color = label_map[tier]
        is_predicted = tier == prediction
        border = f"3px solid {color}" if is_predicted else "1px solid rgba(255,255,255,0.1)"
        bg = f"rgba({','.join(str(int(color.lstrip('#')[i:i+2], 16)) for i in (0,2,4))},0.22)" if is_predicted else "rgba(17,24,39,0.6)"
        scale = "transform: scale(1.06);" if is_predicted else ""
        badge = f'<div style="background:{color};color:#fff;border-radius:8px;padding:2px 10px;font-size:0.75rem;font-weight:700;display:inline-block;margin-bottom:6px;">{"✅ PREDICTED" if is_predicted else f"Range {tier}"}</div>' 
        col.markdown(
            f"""<div style="border:{border};border-radius:16px;padding:16px 12px;text-align:center;background:{bg};{scale}min-height:140px;">
                {badge}<br>
                <span style="font-size:1.8rem;">{icon}</span><br>
                <span style="font-weight:700;font-size:1rem;color:#f1f5f9;">{name}</span><br>
                <span style="font-size:0.82rem;color:#94a3b8;">{price_range}</span>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Key features summary alongside result ----
    st.markdown("### 📋 Features Used for This Prediction")
    feature_groups = {
        "Core Specs": {
            "Battery Power": f"{battery_power} mAh",
            "RAM": f"{ram} MB",
            "No. of Cores": str(n_cores),
            "Clock Speed": f"{clock_speed} GHz",
        },
        "Display & Camera": {
            "Resolution": f"{px_width} × {px_height} px",
            "Screen Size": f"{sc_w} × {sc_h} cm",
            "Front Camera": f"{fc} MP",
            "Primary Camera": f"{pc} MP",
        },
        "Storage & Build": {
            "Internal Memory": f"{int_memory} GB",
            "Talk Time": f"{talk_time} hrs",
            "Mobile Depth": f"{m_dep} cm",
        },
        "Connectivity": {
            "Bluetooth": "Yes" if blue else "No",
            "Dual SIM": "Yes" if dual_sim else "No",
            "3G": "Yes" if three_g else "No",
            "4G": "Yes" if four_g else "No",
            "WiFi": "Yes" if wifi else "No",
            "Touch Screen": "Yes" if touch_screen else "No",
        },
    }

    fg_cols = st.columns(4)
    for col, (group_name, features) in zip(fg_cols, feature_groups.items()):
        rows = "".join(
            f'<tr><td style="color:#94a3b8;padding:3px 6px;">{k}</td>'
            f'<td style="color:#f1f5f9;font-weight:600;padding:3px 6px;">{v}</td></tr>'
            for k, v in features.items()
        )
        col.markdown(
            f"""<div style="background:rgba(17,24,39,0.7);border:1px solid rgba(99,102,241,0.2);border-radius:14px;padding:14px;">
                <div style="font-weight:700;color:#a5b4fc;margin-bottom:8px;font-size:0.9rem;">{group_name}</div>
                <table style="width:100%;border-collapse:collapse;font-size:0.85rem;">{rows}</table>
            </div>""",
            unsafe_allow_html=True,
        )
import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Mobile Price Prediction",
    page_icon="📱",
    layout="wide"
)

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
model = joblib.load("mobile_price_model.pkl")

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

# ---------- Input layout ----------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Core Specs")
    id_val = st.number_input("Mobile ID", min_value=1, value=1, step=1)
    battery_power = st.number_input("Battery Power", min_value=0, value=1500, step=1)
    ram = st.number_input("RAM", min_value=0, value=2500, step=1)
    n_cores = st.number_input("Number of Cores", min_value=1, value=4, step=1)
    clock_speed = st.number_input("Clock Speed", min_value=0.0, value=2.0, step=0.1, format="%.1f")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Display & Camera")
    px_height = st.number_input("Pixel Height", min_value=0, value=800, step=1)
    px_width = st.number_input("Pixel Width", min_value=0, value=1200, step=1)
    sc_h = st.number_input("Screen Height", min_value=0, value=12, step=1)
    sc_w = st.number_input("Screen Width", min_value=0, value=6, step=1)
    fc = st.number_input("Front Camera", min_value=0, value=8, step=1)
    pc = st.number_input("Primary Camera", min_value=0, value=16, step=1)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Connectivity & Storage")
    int_memory = st.number_input("Internal Memory", min_value=0, value=64, step=1)
    talk_time = st.number_input("Talk Time", min_value=0, value=12, step=1)
    m_dep = st.number_input("Mobile Depth", min_value=0.0, value=0.5, step=0.1, format="%.1f")
    mobile_wt = st.selectbox("Mobile Weight Category", options=[1, 2, 3], format_func=lambda x: {1:"Low", 2:"Med", 3:"High"}[x])
    blue = st.selectbox("Bluetooth", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    dual_sim = st.selectbox("Dual SIM", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Additional Features")

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    four_g = st.selectbox("4G", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
with c2:
    three_g = st.selectbox("3G", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
with c3:
    touch_screen = st.selectbox("Touch Screen", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
with c4:
    wifi = st.selectbox("WiFi", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
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
    input_df = pd.DataFrame([{
        "id": id_val,
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

    prediction = model.predict(input_df)[0]

    label_map = {
        0: "Low Cost",
        1: "Medium-Low Cost",
        2: "Medium-High Cost",
        3: "High Cost"
    }

    st.markdown(
        f'<div class="pred-box">✅ Predicted Price Range: <b>{prediction}</b> — {label_map[prediction]}</div>',
        unsafe_allow_html=True
    )

    st.subheader("Input Summary")
    st.dataframe(input_df, use_container_width=True)
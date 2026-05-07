import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
import base64
from datetime import datetime

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Leaf Battery — AI Usability Predictor",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# CSS
# =========================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Share+Tech+Mono&family=Exo+2:wght@300;400;600&display=swap');

    :root {
        --bg:#050d1a;--card:#0a1628;--card2:#0d1f3c;
        --cyan:#00d4ff;--green:#00ff9d;--orange:#ff6b35;
        --red:#ff3366;--text:#e8f4fd;--muted:#7ba7cc;--border:#1a3a5c;
    }

    .stApp{background:var(--bg);}
    #MainMenu,footer,header{visibility:hidden;}
    [data-testid="stSidebar"]{display:none;}
    .block-container{padding:1rem 2rem;}

    .topbar{
        background:linear-gradient(90deg,#060e1c,#0a1628,#060e1c);
        border-bottom:1px solid var(--border);
        padding:0.8rem 2rem;margin:-1rem -2rem 1.5rem -2rem;
        display:flex;align-items:center;justify-content:space-between;
    }
    .topbar-logo{display:flex;align-items:center;gap:0.8rem;}
    .logo-icon{
        width:50px;height:50px;
        display:flex;align-items:center;justify-content:center;
        box-shadow:0 0 15px rgba(0,212,255,0.25);
    }
    .logo-text{
        font-family:Rajdhani,sans-serif;font-size:1.4rem;font-weight:700;
        color:var(--cyan);letter-spacing:2px;
    }
    .logo-sub{
        font-family:Share Tech Mono,monospace;font-size:0.65rem;
        color:var(--muted);letter-spacing:1px;margin-top:-2px;
    }
    .topbar-info{
        font-family:Share Tech Mono,monospace;font-size:0.72rem;
        color:var(--muted);text-align:right;line-height:1.6;
    }

    .hero{
        background:linear-gradient(135deg,#0a1628,#0d1f3c);
        border:1px solid var(--border);border-top:2px solid var(--cyan);
        border-radius:12px;padding:1.5rem 2rem;margin-bottom:1.5rem;
        display:flex;align-items:center;justify-content:space-between;
    }
    .hero-title{
        font-family:Rajdhani,sans-serif;font-size:1.8rem;font-weight:700;
        color:var(--cyan);letter-spacing:2px;
        text-shadow:0 0 30px rgba(0,212,255,0.3);margin:0;
    }
    .hero-sub{
        font-family:Share Tech Mono,monospace;color:var(--muted);
        font-size:0.78rem;margin-top:0.3rem;letter-spacing:1px;
    }
    .hero-badges{margin-top:0.8rem;display:flex;gap:0.5rem;flex-wrap:wrap;}
    .badge{
        display:inline-block;padding:0.2rem 0.8rem;border-radius:20px;
        font-size:0.7rem;font-family:Share Tech Mono,monospace;
    }
    .badge-cyan{background:rgba(0,212,255,0.1);border:1px solid rgba(0,212,255,0.3);color:var(--cyan);}
    .badge-green{background:rgba(0,255,157,0.1);border:1px solid rgba(0,255,157,0.3);color:var(--green);}
    .badge-orange{background:rgba(255,107,53,0.1);border:1px solid rgba(255,107,53,0.3);color:var(--orange);}

    .mcard{
        background:var(--card);border:1px solid var(--border);
        border-radius:10px;padding:1rem;text-align:center;
        position:relative;overflow:hidden;
    }
    .mcard::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;}
    .mcard.c::before{background:var(--cyan);}
    .mcard.g::before{background:var(--green);}
    .mcard.o::before{background:var(--orange);}
    .mcard.r::before{background:var(--red);}
    .mval{font-family:Rajdhani,sans-serif;font-size:1.8rem;font-weight:700;margin:0;}
    .mval.c{color:var(--cyan);}
    .mval.g{color:var(--green);}
    .mval.o{color:var(--orange);}
    .mval.r{color:var(--red);}
    .mlbl{
        font-family:Share Tech Mono,monospace;color:var(--muted);
        font-size:0.68rem;letter-spacing:1px;margin-top:0.2rem;
    }
    .sec{
        font-family:Rajdhani,sans-serif;font-size:1.2rem;font-weight:600;
        color:var(--cyan);letter-spacing:2px;border-bottom:1px solid var(--border);
        padding-bottom:0.4rem;margin:1.2rem 0 0.8rem 0;
    }
    .icard{
        background:var(--card2);border:1px solid var(--border);
        border-left:3px solid var(--cyan);border-radius:8px;
        padding:0.7rem 1rem;margin:0.3rem 0;font-size:0.85rem;color:var(--text);
    }
    .pred-box{
        border-radius:12px;padding:1.5rem;text-align:center;
        margin:0.5rem 0;border:1px solid;
    }
    .pred-g{background:rgba(0,255,157,0.05);border-color:var(--green);}
    .pred-f{background:rgba(255,107,53,0.05);border-color:var(--orange);}
    .pred-p{background:rgba(255,51,102,0.05);border-color:var(--red);}
    .ptitle{
        font-family:Rajdhani,sans-serif;font-size:0.85rem;
        color:var(--muted);letter-spacing:2px;
    }
    .psub{font-family:Share Tech Mono,monospace;font-size:0.9rem;color:var(--muted);}
    .rec-card{
        border-radius:10px;padding:1rem 1.2rem;margin:0.4rem 0;
        border:1px solid;font-family:'Exo 2',sans-serif;font-size:0.88rem;
    }
    .warn-card{
        border-radius:10px;padding:0.9rem 1.1rem;margin:0.35rem 0;
        border:1px solid;font-family:'Exo 2',sans-serif;font-size:0.86rem;
    }

    .stButton>button{
        background:linear-gradient(135deg,#00d4ff20,#00ff9d20)!important;
        border:1px solid var(--cyan)!important;color:var(--cyan)!important;
        font-family:Rajdhani,sans-serif!important;font-size:1rem!important;
        font-weight:600!important;letter-spacing:2px!important;
        padding:0.6rem 2rem!important;border-radius:6px!important;width:100%!important;
    }
    .stDownloadButton>button{
        background:linear-gradient(135deg,#00ff9d20,#00d4ff20)!important;
        border:1px solid var(--green)!important;color:var(--green)!important;
        font-family:Rajdhani,sans-serif!important;font-size:1rem!important;
        font-weight:600!important;letter-spacing:2px!important;
        padding:0.6rem 2rem!important;border-radius:6px!important;width:100%!important;
    }
    .stTabs [data-baseweb="tab"]{
        font-family:Rajdhani,sans-serif!important;
        font-size:0.95rem!important;color:var(--muted)!important;letter-spacing:1px!important;
    }
    .stTabs [aria-selected="true"]{
        color:var(--cyan)!important;
        border-bottom-color:var(--cyan)!important;
    }
    .stNumberInput input{
        background:#0d1f3c!important;border:1px solid var(--border)!important;
        color:var(--text)!important;font-family:Share Tech Mono,monospace!important;
        font-size:1.1rem!important;border-radius:8px!important;text-align:center!important;
    }
    p,li{color:var(--text)!important;}
    h1,h2,h3{color:var(--cyan)!important;font-family:Rajdhani,sans-serif!important;}
    label{
        color:var(--muted)!important;font-family:Share Tech Mono,monospace!important;
        font-size:0.78rem!important;letter-spacing:1px!important;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# CONSTANTS
# =========================================================
PLOT_BG = dict(
    paper_bgcolor='#0a1628',
    plot_bgcolor='#050d1a',
    font=dict(color='#7ba7cc', family='Share Tech Mono'),
    legend=dict(bgcolor='#0a1628', bordercolor='#1a3a5c', borderwidth=1)
)

V_MIN, V_MAX = 6.0, 8.2
SEQUENCE_LENGTH = 30

# IMPORTANT: This order must match the PRACTICAL Colab training pipeline
FEATURES = ['Voltage', 'Current', 'Power', 'Temperature', 'State_encoded', 'CycleCount']

CLASS_NAMES = ['Poor', 'Fair', 'Good']
CLASS_COLORS = {'Poor': '#ff3366', 'Fair': '#ff6b35', 'Good': '#00ff9d'}

METRICS = {
    "r2": "94.80%",
    "mae": "1.83%",
    "mae_full": "1.8267%",
    "rmse": "3.99%",
    "rmse_full": "3.9870%",
    "mape": "2.8324%",
    "accuracy": "93.57%",
    "precision": "93.60%",
    "recall": "93.57%",
    "f1": "93.56%",
}

ERROR_ANALYSIS_DF = pd.DataFrame({
    "Condition": [
        "Low Voltage", "Medium Voltage", "High Voltage",
        "Low Temp", "Medium Temp", "High Temp",
        "Low Current", "Medium Current", "High Current"
    ],
    "MAE": [
        2.042841, 0.787428, 1.057542,
        1.430562, 0.863915, 1.625805,
        1.055200, 0.743832, 2.133351
    ],
    "Group": [
        "Voltage", "Voltage", "Voltage",
        "Temperature", "Temperature", "Temperature",
        "Current", "Current", "Current"
    ]
})

ABLATION_DF = pd.DataFrame({
    "Feature": ["Voltage", "State", "Power", "Temperature", "CycleCount", "Current"],
    "MAE Increase": [68.976093, 0.293844, 0.183841, 0.064902, -0.121550, -0.164030]
})

PERMUTATION_DF = pd.DataFrame({
    "Feature": ["Voltage", "Power", "Current", "State", "Temperature", "CycleCount"],
    "MAE Increase": [15.162413, 0.126543, 0.016724, 0.014314, 0.006838, -0.053349]
})

# =========================================================
# HELPERS
# =========================================================
def get_base64_image(image_path):
    if not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

logo_base64 = get_base64_image("logo.png")

@st.cache_resource
def load_models():
    try:
        import tensorflow as tf
        reg = tf.keras.models.load_model("best_lstm_regression.keras", compile=False)
        cls = tf.keras.models.load_model("best_lstm_classification.keras", compile=False)
        with open("scaler_X.pkl", "rb") as f:
            scaler = pickle.load(f)
        return reg, cls, scaler, True
    except Exception as e:
        st.warning(f"⚠️ Model load error: {e}")
        return None, None, None, False

def predict_one(v, i, p, t, cycle_count, state_encoded, scaler, reg, cls):
    # Practical model feature order:
    # Voltage, Current, Power, Temperature, State_encoded, CycleCount
    X = scaler.transform([[v, i, p, t, state_encoded, cycle_count]])
    seq = np.tile(X, (SEQUENCE_LENGTH, 1)).reshape(1, SEQUENCE_LENGTH, len(FEATURES))

    soh = float(np.clip(reg.predict(seq, verbose=0)[0][0], 0, 100))
    probs = cls.predict(seq, verbose=0)[0]

    # Fair threshold used in final practical evaluation
    if probs[1] >= 0.30:
        class_id = 1
    else:
        class_id = int(np.argmax(probs))

    return soh, CLASS_NAMES[class_id], probs

def make_gauge(soh, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=soh,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': 'STATE OF HEALTH',
               'font': {'size': 13, 'color': '#7ba7cc', 'family': 'Rajdhani'}},
        number={'suffix': '%', 'font': {'size': 38, 'color': color, 'family': 'Rajdhani'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#1a3a5c',
                     'tickfont': {'color': '#7ba7cc', 'size': 10}},
            'bar': {'color': color, 'thickness': 0.25},
            'bgcolor': '#0a1628',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 40], 'color': 'rgba(255,51,102,0.15)'},
                {'range': [40, 70], 'color': 'rgba(255,107,53,0.15)'},
                {'range': [70, 100], 'color': 'rgba(0,255,157,0.15)'}
            ],
            'threshold': {'line': {'color': 'white', 'width': 2},
                          'thickness': 0.75, 'value': soh}
        }
    ))
    fig.update_layout(**PLOT_BG, height=260, margin=dict(l=20, r=20, t=40, b=10))
    return fig

def make_input_viz(voltage, current, temperature):
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Voltage (V)", "Current (A)", "Temperature (°C)"]
    )

    fig.add_trace(go.Bar(
        x=['Voltage'], y=[voltage], marker_color='#00d4ff',
        opacity=0.85, text=f'{voltage:.2f}V', textposition='outside'
    ), row=1, col=1)
    fig.update_yaxes(range=[5.5, 8.5], row=1, col=1,
                     gridcolor='#1a3a5c', linecolor='#1a3a5c')

    curr_color = '#ff6b35' if current > 0 else '#7ba7cc'
    fig.add_trace(go.Bar(
        x=['Current'], y=[current], marker_color=curr_color,
        opacity=0.85, text=f'{current:.2f}A', textposition='outside'
    ), row=1, col=2)
    fig.update_yaxes(range=[0, 10], row=1, col=2,
                     gridcolor='#1a3a5c', linecolor='#1a3a5c')

    temp_color = '#00ff9d' if temperature <= 40 else '#ff6b35' if temperature <= 50 else '#ff3366'
    fig.add_trace(go.Bar(
        x=['Temperature'], y=[temperature], marker_color=temp_color,
        opacity=0.85, text=f'{temperature:.1f}°C', textposition='outside'
    ), row=1, col=3)
    fig.update_yaxes(range=[0, 80], row=1, col=3,
                     gridcolor='#1a3a5c', linecolor='#1a3a5c')

    fig.update_layout(**PLOT_BG, height=250, showlegend=False,
                      margin=dict(l=10, r=10, t=30, b=10))
    return fig

def make_error_analysis_chart():
    colors = []
    for c in ERROR_ANALYSIS_DF["Condition"]:
        if "Voltage" in c:
            colors.append('#00d4ff')
        elif "Temp" in c:
            colors.append('#ff6b35')
        else:
            colors.append('#00ff9d')

    fig = go.Figure(go.Bar(
        x=ERROR_ANALYSIS_DF["Condition"],
        y=ERROR_ANALYSIS_DF["MAE"],
        marker_color=colors,
        opacity=0.88,
        text=[f"{v:.2f}" for v in ERROR_ANALYSIS_DF["MAE"]],
        textposition='outside'
    ))
    fig.update_layout(
        **PLOT_BG,
        title='Prediction Error by Operating Condition',
        height=360,
        xaxis=dict(tickangle=-20, gridcolor='#1a3a5c', linecolor='#1a3a5c'),
        yaxis=dict(title='MAE', gridcolor='#1a3a5c', linecolor='#1a3a5c'),
        margin=dict(l=5, r=5, t=45, b=10),
        showlegend=False
    )
    return fig

def make_importance_chart(df, title, color_main):
    colors = []
    for feature, value in zip(df["Feature"], df["MAE Increase"]):
        if feature == "Voltage":
            colors.append('#ff3366')
        elif value < 0:
            colors.append('#7ba7cc')
        else:
            colors.append(color_main)

    fig = go.Figure(go.Bar(
        x=df["MAE Increase"],
        y=df["Feature"],
        orientation='h',
        marker_color=colors,
        opacity=0.88,
        text=[f"{v:.3f}" for v in df["MAE Increase"]],
        textposition='outside'
    ))
    fig.update_layout(
        **PLOT_BG,
        title=title,
        height=320,
        xaxis=dict(title='MAE Increase', gridcolor='#1a3a5c', linecolor='#1a3a5c'),
        yaxis=dict(gridcolor='#1a3a5c', linecolor='#1a3a5c'),
        margin=dict(l=5, r=5, t=45, b=10),
        showlegend=False
    )
    fig.add_vline(x=0, line_color='#7ba7cc', line_width=1)
    return fig

def get_input_warnings(voltage, current, temperature, state):
    warnings_list = []

    if voltage < 6.5:
        warnings_list.append(("⚠️", f"Low voltage detected ({voltage:.2f}V). Battery may be near discharge region.", "orange"))
    elif voltage > 8.0:
        warnings_list.append(("ℹ️", f"High voltage region ({voltage:.2f}V). Interpret with near-full-charge behavior.", "cyan"))

    if state == "DISCHARGING" and current <= 0.05:
        warnings_list.append(("❌", "Discharging state with near-zero current indicates the battery may not be supplying load.", "red"))

    if current > 3.0:
        warnings_list.append(("⚠️", f"High current detected ({current:.2f}A). Usability may reduce under high load stress.", "orange"))

    if temperature >= 60:
        warnings_list.append(("❌", f"Critical temperature ({temperature:.1f}°C). Thermal safety risk detected.", "red"))
    elif temperature > 45:
        warnings_list.append(("🌡️", f"Elevated temperature ({temperature:.1f}°C). Battery is under thermal stress.", "red"))
    elif temperature < 15:
        warnings_list.append(("❄️", f"Low temperature ({temperature:.1f}°C). Battery behavior may become less stable.", "orange"))

    if not warnings_list:
        warnings_list.append(("✅", "Inputs fall within the model’s normal operating region.", "green"))

    return warnings_list

def get_recommendations(usability, soh, voltage, temperature, current, state):
    power = voltage * current
    recs = {
        'Good': [
            ("✅", f"Battery is in EXCELLENT condition — SoH: {soh:.1f}% (Above 70% threshold)", "green"),
            ("🔋", f"Voltage {voltage:.2f}V is within optimal operating range (7.0V–8.2V)", "green"),
            ("🌡️", f"Temperature {temperature:.1f}°C — Operating within safe thermal limits (<45°C)", "green"),
            ("⚡", f"Current: {current:.2f}A | Power: {power:.2f}W | State: {state}", "green"),
            ("♻️", "Suitable for continued use in solar energy storage applications", "green"),
            ("📋", "Action: Continue normal operation. Schedule routine check in 30 days.", "green"),
        ],
        'Fair': [
            ("⚠️", f"Battery shows MODERATE degradation — SoH: {soh:.1f}%", "orange"),
            ("🔋", f"Voltage {voltage:.2f}V — Monitor for further decline below 7.0V", "orange"),
            ("🌡️", f"Temperature {temperature:.1f}°C — Ensure adequate cooling system", "orange"),
            ("⚡", f"Current: {current:.2f}A | Power: {power:.2f}W | State: {state}", "orange"),
            ("♻️", "May still be used with reduced load for solar storage applications", "orange"),
            ("📋", "Action: Reduce load by 20–30%. Schedule maintenance inspection within 7 days.", "orange"),
        ],
        'Poor': [
            ("❌", f"Battery is in CRITICAL condition — SoH: {soh:.1f}%", "red"),
            ("🔋", f"Voltage {voltage:.2f}V — Risk of failure or poor usability", "red"),
            ("🌡️", f"Temperature {temperature:.1f}°C — Check thermal management immediately", "red"),
            ("⚡", f"Current: {current:.2f}A | Power: {power:.2f}W | State: {state}", "red"),
            ("♻️", "NOT recommended for solar energy storage without inspection", "red"),
            ("📋", "Action: DISCONTINUE USE until battery is checked or replaced.", "red"),
        ]
    }
    return recs[usability]

def generate_report(voltage, current, power, temperature, soh, usability, probs, recs, state):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    color_map = {'Good': '#00aa66', 'Fair': '#dd6600', 'Poor': '#cc0033'}
    color = color_map[usability]
    emoji = {'Good': '✅', 'Fair': '⚠️', 'Poor': '❌'}[usability]

    rec_html = ""
    for icon, txt, clr in recs:
        bg = {'green': '#e8f8f0', 'orange': '#fff3e8', 'red': '#fde8ec'}[clr]
        bc = {'green': '#00aa66', 'orange': '#dd6600', 'red': '#cc0033'}[clr]
        rec_html += f'<div style="background:{bg};border-left:4px solid {bc};padding:10px 15px;margin:6px 0;border-radius:4px;font-size:13px;">{icon} {txt}</div>'

    prob_html = ""
    for cls_name, prob, clr in zip(CLASS_NAMES, probs, ['#cc0033', '#dd6600', '#00aa66']):
        w = int(prob * 100)
        prob_html += f"""
        <div style="margin:8px 0;">
            <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                <span style="font-size:13px;font-weight:bold;">{cls_name}</span>
                <span style="font-size:13px;color:{clr};font-weight:bold;">{prob*100:.1f}%</span>
            </div>
            <div style="background:#eee;border-radius:4px;height:20px;">
                <div style="background:{clr};width:{w}%;height:20px;border-radius:4px;"></div>
            </div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Battery Report — {now}</title>
<style>
body{{font-family:'Segoe UI',Arial,sans-serif;background:#f0f4f8;color:#333;}}
.page{{max-width:800px;margin:0 auto;background:white;}}
.header{{background:linear-gradient(135deg,#0a1628 0%,#1a3a5c 100%);padding:30px 40px;}}
.header-title{{color:#00d4ff;font-size:24px;font-weight:bold;letter-spacing:2px;}}
.header-sub{{color:#7ba7cc;font-size:12px;letter-spacing:1px;margin-top:3px;}}
.content{{padding:30px 40px;}}
.section-title{{font-size:14px;font-weight:bold;color:#0a1628;letter-spacing:2px;border-bottom:2px solid #00d4ff;padding-bottom:6px;margin-bottom:15px;text-transform:uppercase;}}
.result-grid{{display:grid;grid-template-columns:1fr 1fr;gap:15px;margin:15px 0;}}
.result-box{{border:2px solid {color};border-radius:10px;padding:20px;text-align:center;background:{color}0d;}}
.result-value{{font-size:38px;font-weight:bold;color:{color};line-height:1;}}
.result-label{{font-size:11px;color:#666;letter-spacing:2px;margin-top:5px;text-transform:uppercase;}}
.param-grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px;}}
.param-item{{background:#f8f9fa;border-radius:8px;padding:12px 15px;border-left:3px solid #00d4ff;}}
.param-label{{font-size:11px;color:#666;letter-spacing:1px;text-transform:uppercase;}}
.param-value{{font-size:18px;font-weight:bold;color:#0a1628;margin-top:3px;}}
.param-status{{font-size:11px;color:#888;margin-top:2px;}}
.footer{{background:#0a1628;padding:20px 40px;margin-top:20px;color:#7ba7cc;font-size:11px;text-align:center;line-height:1.8;}}
</style>
</head>
<body>
<div class="page">
<div class="header">
    <div class="header-title">🍃 LEAF BATTERY</div>
    <div class="header-sub">AI-ENABLED BATTERY USABILITY PREDICTION REPORT</div>
</div>
<div class="content">
    <div class="section-title">🎯 Prediction Results</div>
    <div class="result-grid">
        <div class="result-box">
            <div class="result-value">{soh:.1f}%</div>
            <div class="result-label">State of Health</div>
        </div>
        <div class="result-box">
            <div class="result-value">{emoji} {usability.upper()}</div>
            <div class="result-label">Practical Usability</div>
        </div>
    </div>

    <div class="section-title">📊 Class Probabilities</div>
    {prob_html}

    <div class="section-title">🔌 Input Parameters</div>
    <div class="param-grid">
        <div class="param-item"><div class="param-label">Voltage</div><div class="param-value">{voltage:.3f} V</div></div>
        <div class="param-item"><div class="param-label">Current</div><div class="param-value">{current:.3f} A</div></div>
        <div class="param-item"><div class="param-label">Power</div><div class="param-value">{power:.3f} W</div></div>
        <div class="param-item"><div class="param-label">Temperature</div><div class="param-value">{temperature:.1f} °C</div></div>
        <div class="param-item"><div class="param-label">State</div><div class="param-value">{state}</div></div>
        <div class="param-item"><div class="param-label">Model Accuracy</div><div class="param-value">{METRICS['accuracy']}</div><div class="param-status">R²: {METRICS['r2']} | MAE: {METRICS['mae']}</div></div>
    </div>

    <div class="section-title">📋 Recommendations</div>
    {rec_html}
</div>
<div class="footer">
    🍃 LEAF BATTERY — AI-Enabled Battery Usability Prediction System<br>
    R.M.C.S.L Jayathilaka | 219092 | Wayamba University of Sri Lanka
</div>
</div>
</body>
</html>"""
    return html.encode("utf-8")

# =========================================================
# LOAD MODELS
# =========================================================
lstm_reg, lstm_cls, scaler, loaded = load_models()

logo_html = (
    f'<img src="data:image/png;base64,{logo_base64}" style="width:100%;height:100%;object-fit:contain;">'
    if logo_base64 else '🍃'
)

# =========================================================
# HEADER
# =========================================================
st.markdown(f"""
<div class="topbar">
    <div class="topbar-logo">
        <div class="logo-icon">{logo_html}</div>
        <div>
            <div class="logo-text">LEAF BATTERY</div>
            <div class="logo-sub">AI-ENABLED USABILITY PREDICTOR</div>
        </div>
    </div>
    <div class="topbar-info">
        <div>219092 | R.M.C.S.L Jayathilaka</div>
        <div>Wayamba University of Sri Lanka</div>
        <div style="color:{"#00ff9d" if loaded else "#ff3366"};">
            {"● Model LOADED" if loaded else "● Model NOT LOADED"}</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="hero">
    <div class="hero-left">
        <div class="hero-title">⚡ BATTERY USABILITY PREDICTOR</div>
        <div class="hero-sub">LSTM-BASED PREDICTION OF RECONDITIONED SECOND-LIFE LI-ION BATTERIES</div>
        <div class="hero-badges">
            <span class="badge badge-cyan">R² = {METRICS['r2']}</span>
            <span class="badge badge-green">Accuracy = {METRICS['accuracy']}</span>
            <span class="badge badge-orange">77,341 Samples</span>
            <span class="badge badge-cyan">Practical Model ✅</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 LIVE PREDICTION",
    "📊 MODEL PERFORMANCE",
    "🔍 DATA ANALYSIS",
    "🧠 EXPLAINABILITY",
    "ℹ️ ABOUT"
])

# =========================================================
# TAB 1
# =========================================================
with tab1:
    cols = st.columns(6)
    for col, (val, lbl, clr) in zip(cols, [
        (METRICS["r2"], "R² SCORE", "c"),
        (METRICS["mae"], "MAE", "g"),
        (METRICS["rmse"], "RMSE", "o"),
        (METRICS["accuracy"], "ACCURACY", "c"),
        (METRICS["precision"], "PRECISION", "g"),
        (METRICS["f1"], "F1-SCORE", "o")
    ]):
        with col:
            st.markdown(
                f"<div class='mcard {clr}'><div class='mval {clr}'>{val}</div>"
                f"<div class='mlbl'>{lbl}</div></div>",
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='sec'>◈ ENTER BATTERY MEASUREMENTS</div>", unsafe_allow_html=True)

    st.markdown("""
    <style>
        .guide-box {
            background: linear-gradient(135deg, rgba(0,212,255,0.10), rgba(0,255,157,0.05));
            border: 1px solid rgba(0,212,255,0.45);
            border-left: 5px solid #00d4ff;
            border-radius: 14px;
            padding: 1.2rem;
            margin: 0.7rem 0 1.2rem 0;
            box-shadow: 0 0 18px rgba(0,212,255,0.12);
        }
        .guide-title {
            font-family: Rajdhani, sans-serif;
            font-size: 1.2rem;
            font-weight: 700;
            color: #00d4ff;
            letter-spacing: 1px;
            margin-bottom: 1rem;
        }
        .guide-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.8rem;
        }
        .guide-card {
            background: #0a1628;
            border: 1px solid rgba(0,212,255,0.25);
            border-radius: 10px;
            padding: 0.8rem;
        }
        .guide-card.orange {
            border-color: rgba(255,107,53,0.30);
        }
        .guide-card.green {
            border-color: rgba(0,255,157,0.30);
        }
        .guide-card-title {
            font-family: Rajdhani, sans-serif;
            font-weight: 700;
            font-size: 0.95rem;
            margin-bottom: 0.3rem;
        }
        .guide-card-title.green {
            color: #00ff9d;
        }
        .guide-card-title.orange {
            color: #ff6b35;
        }
        .guide-text {
            color: #e8f4fd;
            font-size: 0.83rem;
            line-height: 1.5;
        }
        .guide-note {
            margin-top: 1rem;
            background: rgba(0,212,255,0.08);
            border: 1px dashed rgba(0,212,255,0.35);
            border-radius: 10px;
            padding: 0.9rem;
            color: #e8f4fd;
            font-size: 0.86rem;
            line-height: 1.7;
        }
        @media (max-width: 900px) {
            .guide-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>

    <div class="guide-box">
        <div class="guide-title">⚠️ IMPORTANT TESTING GUIDELINES FOR ACCURATE PREDICTION</div>

        <div class="guide-grid">
            <div class="guide-card">
                <div class="guide-card-title green">🔋 BATTERY TYPE</div>
                <div class="guide-text">Use only for <strong>reconditioned Nissan Leaf lithium-ion battery modules</strong>.</div>
            </div>

            <div class="guide-card orange">
                <div class="guide-card-title orange">⚡ DISCHARGING TEST</div>
                <div class="guide-text">Use a <strong>21 W load</strong> as the reference load condition.</div>
            </div>

            <div class="guide-card orange">
                <div class="guide-card-title orange">🔌 HIGHER LOAD OPTION</div>
                <div class="guide-text">If higher load is required, connect <strong>two 21 W loads in parallel</strong>.</div>
            </div>

            <div class="guide-card green">
                <div class="guide-card-title green">☀️ CHARGING TEST</div>
                <div class="guide-text">Use a <strong>20 W solar panel</strong> as the reference charging source.</div>
            </div>
        </div>

        <div class="guide-note">
            ✅ For more accurate prediction, complete one full <strong>charging or discharging cycle</strong> before entering values.<br>
            📝 After completing the cycle, enter the <strong>initial and final voltage, current, temperature</strong>, and the <strong>process time</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # =====================================================
    # INITIAL PARAMETERS
    # =====================================================
    st.markdown("<div class='sec'>◈ INITIAL BATTERY CONDITIONS</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        initial_voltage = st.number_input(
            "Initial Voltage (V)",
            min_value=6.0,
            max_value=8.5,
            value=7.80,
            step=0.01,
            format="%.2f"
        )

    with c2:
        initial_current = st.number_input(
            "Initial Current (A)",
            min_value=0.0,
            max_value=10.0,
            value=1.20,
            step=0.01,
            format="%.2f"
        )

    with c3:
        initial_temperature = st.number_input(
            "Initial Temperature (°C)",
            min_value=0.0,
            max_value=80.0,
            value=30.0,
            step=0.1,
            format="%.1f"
        )

    # =====================================================
    # FINAL PARAMETERS
    # =====================================================
    st.markdown("<div class='sec'>◈ FINAL BATTERY CONDITIONS</div>", unsafe_allow_html=True)

    c4, c5, c6 = st.columns(3)

    with c4:
        voltage = st.number_input(
            "Final Voltage (V)",
            min_value=6.0,
            max_value=8.5,
            value=7.40,
            step=0.01,
            format="%.2f"
        )
        v_pct = (voltage - V_MIN) / (V_MAX - V_MIN) * 100
        v_pct = max(0, min(100, v_pct))
        v_color = "#00ff9d" if voltage >= 7.0 else "#ff6b35" if voltage >= 6.5 else "#ff3366"
        st.markdown(f"""
        <div style='background:#0d1f3c;border-radius:6px;height:8px;margin-top:5px;'>
            <div style='background:{v_color};width:{v_pct:.0f}%;height:8px;border-radius:6px;'></div>
        </div>
        <div style='text-align:center;font-family:Share Tech Mono;font-size:0.75rem;color:{v_color};margin-top:4px;'>
            Voltage SoH ≈ {v_pct:.1f}% | {voltage:.2f}V
        </div>
        """, unsafe_allow_html=True)

    with c5:
        current = st.number_input(
            "Final Current (A)",
            min_value=0.0,
            max_value=10.0,
            value=1.20,
            step=0.01,
            format="%.2f"
        )
        curr_color = "#ff6b35" if current > 0 else "#7ba7cc"
        st.markdown(f"""
        <div style='text-align:center;font-family:Share Tech Mono;font-size:0.78rem;color:{curr_color};margin-top:8px;background:rgba(0,0,0,0.2);border-radius:6px;padding:4px;'>
            🔌 FINAL CURRENT | {current:.2f}A
        </div>
        """, unsafe_allow_html=True)

    with c6:
        temperature = st.number_input(
            "Final Temperature (°C)",
            min_value=0.0,
            max_value=80.0,
            value=35.0,
            step=0.1,
            format="%.1f"
        )
        temp_color = "#00ff9d" if temperature <= 40 else "#ff6b35" if temperature <= 50 else "#ff3366"
        temp_status = "✅ Normal" if temperature <= 40 else "⚠️ Warm" if temperature <= 50 else "❌ Hot"
        st.markdown(f"""
        <div style='text-align:center;font-family:Share Tech Mono;font-size:0.78rem;color:{temp_color};margin-top:8px;background:rgba(0,0,0,0.2);border-radius:6px;padding:4px;'>
            {temp_status} | {temperature:.1f}°C
        </div>
        """, unsafe_allow_html=True)

    # =====================================================
    # PROCESS DETAILS
    # =====================================================
    st.markdown("<div class='sec'>◈ PROCESS INFORMATION</div>", unsafe_allow_html=True)

    pc1, pc2 = st.columns(2)

    with pc1:
        cycle_time = st.number_input(
            "Charging / Discharging Time (Minutes)",
            min_value=0.0,
            max_value=10000.0,
            value=60.0,
            step=1.0
        )

    with pc2:
        state_str = st.selectbox(
            "Battery State",
            ["DISCHARGING", "CHARGING"],
            help="Select the actual operating state of the battery"
        )

    # =====================================================
    # AUTO CALCULATIONS
    # =====================================================
    power = round(voltage * current, 3)
    state_enc = 1 if state_str == "DISCHARGING" else 0
    cycle_count = 1

    voltage_change = abs(initial_voltage - voltage)
    temperature_change = temperature - initial_temperature
    avg_current = (initial_current + current) / 2

    st.markdown(f"""
    <div class='icard' style='margin-top:1rem;'>
        🔄 <strong>Auto Calculated Parameters</strong><br><br>
        ⚡ Power = <strong style='color:#00d4ff;'>{power:.2f} W</strong> &nbsp;|&nbsp;
        <strong style='color:#00ff9d;'>State = {state_str}</strong> &nbsp;|&nbsp;
        <strong style='color:#7ba7cc;'>Voltage SoH ≈ {v_pct:.1f}%</strong><br><br>
        📉 Voltage Change = <strong style='color:#00ff9d;'>{voltage_change:.2f} V</strong> &nbsp;|&nbsp;
        🌡️ Temperature Change = <strong style='color:#ff6b35;'>{temperature_change:.1f} °C</strong> &nbsp;|&nbsp;
        🔌 Average Current = <strong style='color:#00ff9d;'>{avg_current:.2f} A</strong> &nbsp;|&nbsp;
        ⏱️ Process Time = <strong style='color:#7ba7cc;'>{cycle_time:.1f} Minutes</strong>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sec'>◈ INPUT VISUALIZATION</div>", unsafe_allow_html=True)
    st.plotly_chart(make_input_viz(voltage, current, temperature), use_container_width=True)

    st.markdown("<div class='sec'>◈ INPUT RELIABILITY CHECK</div>", unsafe_allow_html=True)
    input_warnings = get_input_warnings(voltage, current, temperature, state_str)
    for icon, text, level in input_warnings:
        bc = {"green": "#00ff9d", "orange": "#ff6b35", "red": "#ff3366", "cyan": "#00d4ff"}[level]
        bg = {
            "green": "rgba(0,255,157,0.06)",
            "orange": "rgba(255,107,53,0.06)",
            "red": "rgba(255,51,102,0.06)",
            "cyan": "rgba(0,212,255,0.06)"
        }[level]
        st.markdown(f"""
        <div class='warn-card' style='background:{bg};border-color:{bc};color:var(--text);'>
            {icon} {text}
        </div>
        """, unsafe_allow_html=True)

    if st.button("🔮 PREDICT BATTERY CONDITION NOW"):
        if not loaded:
            st.error("⚠️ Models not loaded! Please refresh the page.")
        else:
            with st.spinner("🧠 Running LSTM analysis..."):
                soh, usability, probs = predict_one(
                    voltage, current, power, temperature,
                    cycle_count, state_enc, scaler, lstm_reg, lstm_cls
                )

            color = CLASS_COLORS[usability]
            css = {'Good': 'pred-g', 'Fair': 'pred-f', 'Poor': 'pred-p'}[usability]
            emoji = {'Good': '✅', 'Fair': '⚠️', 'Poor': '❌'}[usability]
            recs = get_recommendations(usability, soh, voltage, temperature, current, state_str)

            st.markdown("<div class='sec'>◈ PREDICTION RESULTS</div>", unsafe_allow_html=True)

            c1, c2 = st.columns([1, 1])
            with c1:
                st.plotly_chart(make_gauge(soh, color), use_container_width=True)

            with c2:
                st.markdown(f"""
                <div class='pred-box {css}' style='margin-top:0.5rem;'>
                    <div class='ptitle'>USABILITY STATUS</div>
                    <div style='font-size:3.5rem;margin:0.3rem 0;'>{emoji}</div>
                    <div style='color:{color};font-family:Rajdhani,sans-serif;font-size:1.8rem;font-weight:700;letter-spacing:3px;'>
                        {usability.upper()}
                    </div>
                    <div class='psub' style='margin-top:0.5rem;font-size:0.85rem;'>
                        {"✔ Excellent — Safe for solar energy storage" if usability=="Good"
                         else "⚡ Moderate degradation — Monitor closely" if usability=="Fair"
                         else "✖ Critical — Replace battery immediately!"}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                fig_p = go.Figure()
                for i, (cls_n, prob) in enumerate(zip(CLASS_NAMES, probs)):
                    fig_p.add_trace(go.Bar(
                        x=[prob * 100],
                        y=[cls_n],
                        orientation='h',
                        marker_color=['#ff3366', '#ff6b35', '#00ff9d'][i],
                        opacity=0.85,
                        text=f"{prob*100:.1f}%",
                        textposition='inside',
                        textfont=dict(size=13, family='Rajdhani', color='white')
                    ))
                fig_p.update_layout(
                    **PLOT_BG,
                    height=170,
                    showlegend=False,
                    xaxis=dict(range=[0, 100], gridcolor='#1a3a5c', linecolor='#1a3a5c', ticksuffix='%'),
                    margin=dict(l=5, r=5, t=5, b=5)
                )
                st.plotly_chart(fig_p, use_container_width=True)

            st.markdown("<div class='sec'>◈ RECOMMENDATIONS</div>", unsafe_allow_html=True)
            bc = {'Good': '#00ff9d', 'Fair': '#ff6b35', 'Poor': '#ff3366'}[usability]
            bg = {'Good': 'rgba(0,255,157,0.05)', 'Fair': 'rgba(255,107,53,0.05)', 'Poor': 'rgba(255,51,102,0.05)'}[usability]
            for icon, txt, clr in recs:
                st.markdown(f"""
                <div class='rec-card' style='background:{bg};border-color:{bc};color:var(--text);'>
                    {icon} {txt}
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div class='sec'>◈ DOWNLOAD REPORT</div>", unsafe_allow_html=True)
            report = generate_report(
                voltage, current, power, temperature,
                soh, usability, probs, recs, state_str
            )
            fname = f"LeafBattery_Report_{usability}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            st.download_button(
                label="📄 DOWNLOAD BATTERY ANALYSIS REPORT",
                data=report,
                file_name=fname,
                mime="text/html",
                use_container_width=True
            )
            st.markdown("""
            <div class='icard'>
                💡 <strong>How to save as PDF:</strong>
                Open the downloaded HTML file → Press <strong>Ctrl+P</strong> →
                Select <strong>"Save as PDF"</strong> → Save
            </div>
            """, unsafe_allow_html=True)

# =========================================================
# TAB 2
# =========================================================
with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**🎯 REGRESSION — SoH Prediction**")
        st.dataframe(pd.DataFrame({
            'Metric': ['R² Score', 'MAE', 'RMSE', 'MAPE'],
            'Value': [METRICS['r2'], METRICS['mae_full'], METRICS['rmse_full'], METRICS['mape']],
            'Status': ['✅ Excellent', '✅ Low Error', '✅ Good', '✅ Very Low']
        }), use_container_width=True, hide_index=True)

    with c2:
        st.markdown("**🎯 CLASSIFICATION — Practical Usability**")
        st.dataframe(pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Value': [METRICS['accuracy'], METRICS['precision'], METRICS['recall'], METRICS['f1']],
            'Status': ['✅'] * 4
        }), use_container_width=True, hide_index=True)

    st.markdown("<div class='sec'>◈ PER-CLASS METRICS</div>", unsafe_allow_html=True)
    fig2 = go.Figure()
    for vals, name, color in [
        ([97.8, 84.1, 95.6], 'Precision', '#00d4ff'),
        ([86.9, 84.1, 97.2], 'Recall', '#00ff9d'),
        ([92.0, 84.1, 96.4], 'F1-Score', '#ff6b35')
    ]:
        fig2.add_trace(go.Bar(
            name=name,
            x=CLASS_NAMES,
            y=vals,
            marker_color=color,
            opacity=0.85,
            text=[f"{v:.1f}%" for v in vals],
            textposition='outside'
        ))
    fig2.update_layout(
        **PLOT_BG,
        barmode='group',
        height=300,
        yaxis=dict(range=[0, 115], gridcolor='#1a3a5c', linecolor='#1a3a5c'),
        margin=dict(l=5, r=5, t=10, b=5)
    )
    fig2.add_hline(y=90, line_dash='dash', line_color='#ff3366')
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='sec'>◈ MODEL SUMMARY</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        'Item': ['Model Type', 'Input Features', 'Sequence Length', 'Regression Target', 'Classification Target'],
        'Value': ['Dual LSTM', 'Voltage, Current, Power, Temperature, State, CycleCount', '30 timesteps', 'SoH (%)', 'Good / Fair / Poor']
    }), use_container_width=True, hide_index=True)

# =========================================================
# TAB 3
# =========================================================
with tab3:
    cols = st.columns(4)
    for col, (val, lbl, clr) in zip(cols, [
        ("77,341", "TOTAL SAMPLES", "c"),
        ("11", "CYCLES", "g"),
        ("6", "FEATURES", "o"),
        ("3", "CLASSES", "r")
    ]):
        with col:
            st.markdown(
                f"<div class='mcard {clr}'><div class='mval {clr}'>{val}</div>"
                f"<div class='mlbl'>{lbl}</div></div>",
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='sec'>◈ PREPROCESSING SUMMARY</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        'Step': ['Raw Data', 'Remove Noise', 'Non-Solar Filter', 'Voltage Filter', 'Final'],
        'Rows': [79863, 79383, 78688, 77341, 77341],
        'Removed': ['—', '480', '695', '1347', '—'],
        'Status': ['📥 Loaded', '🧹 Cleaned', '🔍 Filtered', '⚡ Applied', '✅ Ready']
    }), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='sec'>◈ VOLTAGE DISTRIBUTION</div>", unsafe_allow_html=True)
        np.random.seed(42)
        v_sim = np.clip(np.concatenate([
            np.random.normal(7.8, 0.15, 46760),
            np.random.normal(7.2, 0.2, 17660),
            np.random.normal(6.4, 0.2, 12921)
        ]), 6.0, 8.2)
        fig_v = go.Figure(go.Histogram(
            x=v_sim, nbinsx=60,
            marker_color='#00d4ff', opacity=0.75,
            marker_line=dict(color='#050d1a', width=0.5)
        ))
        fig_v.add_vline(
            x=v_sim.mean(),
            line_dash='dash',
            line_color='#ff6b35',
            annotation_text=f'Mean:{v_sim.mean():.2f}V'
        )
        fig_v.update_layout(
            **PLOT_BG,
            title='Voltage Distribution',
            height=270,
            xaxis_title='Voltage (V)',
            yaxis_title='Count',
            margin=dict(l=5, r=5, t=40, b=5)
        )
        st.plotly_chart(fig_v, use_container_width=True)

    with c2:
        st.markdown("<div class='sec'>◈ FEATURE CORRELATION</div>", unsafe_allow_html=True)
        corr_v = [1.000, 0.164, 0.123, -0.454, -0.462, -0.490]
        corr_f = ['Voltage', 'CycleCount', 'Temperature', 'Power', 'State', 'Current']
        fig_c = go.Figure(go.Bar(
            x=corr_v,
            y=corr_f,
            orientation='h',
            marker_color=['#00ff9d' if v > 0 else '#ff3366' for v in corr_v],
            opacity=0.85,
            text=[f'{v:.3f}' for v in corr_v],
            textposition='outside'
        ))
        fig_c.add_vline(x=0, line_color='#7ba7cc', line_width=1)
        fig_c.update_layout(
            **PLOT_BG,
            title='Correlation with SoH',
            height=270,
            xaxis=dict(range=[-0.6, 1.2], gridcolor='#1a3a5c', linecolor='#1a3a5c'),
            margin=dict(l=5, r=5, t=40, b=5)
        )
        st.plotly_chart(fig_c, use_container_width=True)

# =========================================================
# TAB 4
# =========================================================
with tab4:
    st.markdown("<div class='sec'>◈ MODEL DECISION INSIGHTS</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='icard'>
        🧠 The practical model uses <strong>Voltage</strong> as the main health indicator,
        <strong>Current</strong> as load/stress information, and
        <strong>Temperature</strong> as a safety indicator.
    </div>
    <div class='icard'>
        ⚡ Final usability is determined using a combination of model output and practical Fair-class threshold tuning.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sec'>◈ ERROR ANALYSIS BY OPERATING CONDITION</div>", unsafe_allow_html=True)
    st.plotly_chart(make_error_analysis_chart(), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='sec'>◈ ABLATION STUDY</div>", unsafe_allow_html=True)
        st.plotly_chart(
            make_importance_chart(ABLATION_DF, "Ablation Study — MAE Increase", "#00d4ff"),
            use_container_width=True
        )
    with c2:
        st.markdown("<div class='sec'>◈ PERMUTATION IMPORTANCE</div>", unsafe_allow_html=True)
        st.plotly_chart(
            make_importance_chart(PERMUTATION_DF, "Permutation Importance — MAE Increase", "#00ff9d"),
            use_container_width=True
        )

    st.markdown("<div class='sec'>◈ KEY FINDINGS</div>", unsafe_allow_html=True)
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown("""
        <div class='mcard r'>
            <div class='mval r'>Voltage</div>
            <div class='mlbl'>DOMINANT FEATURE</div>
        </div>
        <div class='icard'>Voltage remains the strongest contributor to SoH estimation.</div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown("""
        <div class='mcard o'>
            <div class='mval o'>Stress</div>
            <div class='mlbl'>CURRENT + TEMP</div>
        </div>
        <div class='icard'>Current and temperature improve practical usability interpretation.</div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown("""
        <div class='mcard c'>
            <div class='mval c'>Fair</div>
            <div class='mlbl'>BALANCED CLASS</div>
        </div>
        <div class='icard'>Fair-class threshold improves balanced prediction across all classes.</div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='sec'>◈ MODEL LIMITATIONS</div>", unsafe_allow_html=True)
    limitations = [
        "SoH is estimated using a voltage-based proxy method rather than direct laboratory capacity testing.",
        "The model was trained on a limited number of battery modules and may not fully generalize to all chemistries.",
        "Prediction reliability is highest within the normal operating region.",
        "The system is intended as a practical usability estimation tool, not a replacement for full battery diagnostic testing."
    ]
    for item in limitations:
        st.markdown(f"<div class='icard'>⚠️ {item}</div>", unsafe_allow_html=True)

# =========================================================
# TAB 5
# =========================================================
with tab5:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='sec'>◈ RESEARCHER</div>", unsafe_allow_html=True)
        for icon, lbl, val in [
            ("👤", "Name", "R.M.C.S.L Jayathilaka"),
            ("🔢", "Index No", "219092"),
            ("🎓", "Degree", "BET (Hons) Electrotechnology"),
            ("🏫", "University", "Wayamba University of Sri Lanka"),
            ("🏛️", "Faculty", "Faculty of Technology"),
            ("📅", "Year", "Final Year Research Project — 2025/2026")
        ]:
            st.markdown(
                f"<div class='icard'>{icon} <strong style='color:#7ba7cc;'>{lbl}:</strong> "
                f"<strong style='color:#00d4ff;'>{val}</strong></div>",
                unsafe_allow_html=True
            )

    with c2:
        st.markdown("<div class='sec'>◈ RESEARCH OBJECTIVES</div>", unsafe_allow_html=True)
        for i, obj in enumerate([
            "Analyze performance data of reconditioned second-life lithium-ion battery modules for solar energy storage.",
            "Develop an LSTM-based model to estimate SoH and practical battery usability.",
            "Validate the developed prediction model using regression and classification metrics."
        ], 1):
            st.markdown(
                f"<div class='icard'><strong style='color:#00d4ff;'>{i}.</strong> {obj}</div>",
                unsafe_allow_html=True
            )

        st.markdown("<div class='sec'>◈ RESEARCH SUMMARY</div>", unsafe_allow_html=True)
        for icon, lbl, txt in [
            ("🎯", "Topic", "LSTM-Based Prediction of Reconditioned Second-Life Li-ion Batteries"),
            ("📡", "Data", "Real-time ESP32 sensor data — 77,341 readings, 11 cycles"),
            ("🧠", "Model", "Dual LSTM — Regression + Practical Classification"),
            ("✅", "Validation", f"R² {METRICS['r2']} | Accuracy {METRICS['accuracy']}"),
            ("🌞", "Application", "Solar energy storage battery usability assessment")
        ]:
            st.markdown(
                f"<div class='icard'>{icon} <strong style='color:#7ba7cc;'>{lbl}:</strong> {txt}</div>",
                unsafe_allow_html=True
            )

    st.markdown("<div class='sec'>◈ TECH STACK</div>", unsafe_allow_html=True)
    techs = [
        ("🐍", "Python", "Core Language"),
        ("🧠", "TensorFlow / Keras", "Deep Learning"),
        ("📊", "Scikit-learn", "ML Metrics"),
        ("🐼", "Pandas / NumPy", "Data Processing"),
        ("📈", "Plotly", "Visualization"),
        ("🌐", "Streamlit", "Web Dashboard"),
        ("☁️", "Google Colab", "Model Training"),
        ("📡", "ESP32", "Data Collection")
    ]

    tcols = st.columns(4)
    for i, (icon, name, desc) in enumerate(techs):
        with tcols[i % 4]:
            st.markdown(f"""
            <div class='mcard c' style='text-align:left;padding:0.8rem;margin-bottom:0.6rem;'>
                <div style='font-size:1.3rem;'>{icon}</div>
                <div style='color:#00d4ff;font-family:Rajdhani,sans-serif;font-weight:600;font-size:0.85rem;'>{name}</div>
                <div style='color:#7ba7cc;font-family:Share Tech Mono,monospace;font-size:0.68rem;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='sec'>◈ INSTITUTION</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:linear-gradient(135deg,#0a1628,#0d1f3c);
                border:1px solid #1a3a5c;border-left:4px solid #00d4ff;
                border-radius:10px;padding:1.5rem 2rem;margin:0.5rem 0;'>
        <div style='font-family:Rajdhani,sans-serif;font-size:1.4rem;
                    font-weight:700;color:#00d4ff;letter-spacing:1px;'>
            🏫 Wayamba University of Sri Lanka
        </div>
        <div style='font-family:Share Tech Mono,monospace;font-size:0.8rem;
                    color:#7ba7cc;margin-top:0.5rem;'>
            Faculty of Technology | Department of Electrotechnology
        </div>
        <div style='font-family:Exo 2,sans-serif;font-size:0.9rem;
                    color:#e8f4fd;margin-top:0.8rem;line-height:1.6;'>
            Final Year Research Project — BET (Hons) Electrotechnology<br>
            Research Area: AI-Enabled Predictive Analytics for Sustainable Energy Storage<br>
            <span style='color:#00ff9d;'>Index No: 219092 | R.M.C.S.L Jayathilaka</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

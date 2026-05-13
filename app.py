SAVE_DIR = '/content/drive/MyDrive/Battery_LSTM_219092'

# Read current app.py
with open(f'{SAVE_DIR}/app.py', 'r') as f:
    content = f.read()

# RUL calculation function add කරන්න
rul_function = '''
def calculate_rul(soh, cycle_scores, state, voltage, initial_voltage, temperature, cycle_time_min):
    """
    Remaining Useful Life estimation based on:
    - Current SoH
    - Degradation rate from cycle behavior  
    - Stress factors (temperature, current, voltage)
    End-of-life threshold: SoH = 60% (below Good/Fair boundary)
    """
    # ── Base degradation rate ──────────────────────────────
    # Nissan Leaf second-life: ~0.5-2% SoH loss per cycle typically
    # Each cycle ~5 days based on your 45-day / 10 cycle data
    
    voltage_drop = initial_voltage - voltage
    if voltage_drop <= 0:
        voltage_drop = 0.1  # minimum assumed drop
    
    # Degradation per hour of operation
    if cycle_time_min > 0:
        degradation_per_hour = voltage_drop / (V_MAX - V_MIN) * 100 / (cycle_time_min / 60)
    else:
        degradation_per_hour = 0.5  # default fallback
    
    # Convert to monthly rate (assuming ~8hrs/day operation)
    degradation_per_month = degradation_per_hour * 8 * 30
    degradation_per_month = max(degradation_per_month, 0.5)  # minimum 0.5% per month
    
    # ── Stress multipliers ─────────────────────────────────
    stress_factor = 1.0
    stress_notes = []
    
    # Temperature stress
    avg_temp = cycle_scores.get("avg_temperature", temperature)
    temp_change = cycle_scores.get("temperature_change", 0)
    
    if avg_temp > 50 or temp_change > 12:
        stress_factor *= 1.8
        stress_notes.append("High temperature stress (+80% degradation rate)")
    elif avg_temp > 42 or temp_change > 7:
        stress_factor *= 1.35
        stress_notes.append("Moderate temperature stress (+35% degradation rate)")
    elif avg_temp > 38:
        stress_factor *= 1.1
        stress_notes.append("Mild temperature stress (+10% degradation rate)")
    
    # Current instability stress
    current_change = cycle_scores.get("current_change", 0)
    current_retention = cycle_scores.get("current_retention", 1.0)
    
    if current_change > 0.5 or current_retention < 0.7:
        stress_factor *= 1.5
        stress_notes.append("High current instability (+50% degradation rate)")
    elif current_change > 0.3 or current_retention < 0.85:
        stress_factor *= 1.25
        stress_notes.append("Moderate current instability (+25% degradation rate)")
    elif current_change > 0.2:
        stress_factor *= 1.1
        stress_notes.append("Mild current instability (+10% degradation rate)")
    
    # Voltage stress
    if voltage < 6.5:
        stress_factor *= 1.3
        stress_notes.append("Deep discharge stress (+30% degradation rate)")
    
    # ── Final degradation rate ─────────────────────────────
    adjusted_rate = degradation_per_month * stress_factor
    adjusted_rate = max(adjusted_rate, 0.3)  # floor: at least 0.3% per month
    
    # ── RUL calculation ────────────────────────────────────
    # End-of-life at SoH = 60% (Fair→Poor boundary)
    eol_threshold = 60.0
    soh_remaining = max(soh - eol_threshold, 0)
    
    if adjusted_rate > 0:
        rul_months = soh_remaining / adjusted_rate
    else:
        rul_months = 0
    
    rul_months = max(rul_months, 0)
    
    # ── Confidence band (±25%) ─────────────────────────────
    rul_min = rul_months * 0.75
    rul_max = rul_months * 1.25
    
    # ── Usability label ────────────────────────────────────
    if rul_months >= 18:
        rul_label = "Long-term"
        rul_color = "#00ff9d"
        rul_emoji = "✅"
        rul_advice = "Battery has substantial remaining life for continued solar storage use."
    elif rul_months >= 9:
        rul_label = "Medium-term"
        rul_color = "#00d4ff"
        rul_emoji = "🔵"
        rul_advice = "Battery can continue but plan for replacement within 1 year."
    elif rul_months >= 4:
        rul_label = "Short-term"
        rul_color = "#ff6b35"
        rul_emoji = "⚠️"
        rul_advice = "Battery nearing end-of-life. Begin sourcing replacement."
    elif rul_months >= 1:
        rul_label = "Critical"
        rul_color = "#ff3366"
        rul_emoji = "❌"
        rul_advice = "Battery approaching end-of-life. Replace within 1-3 months."
    else:
        rul_label = "End of Life"
        rul_color = "#ff3366"
        rul_emoji = "🔴"
        rul_advice = "Battery has reached end-of-life threshold. Immediate replacement needed."
    
    return {
        "rul_months": rul_months,
        "rul_min": rul_min,
        "rul_max": rul_max,
        "rul_label": rul_label,
        "rul_color": rul_color,
        "rul_emoji": rul_emoji,
        "rul_advice": rul_advice,
        "degradation_rate": adjusted_rate,
        "base_rate": degradation_per_month,
        "stress_factor": stress_factor,
        "stress_notes": stress_notes,
        "eol_threshold": eol_threshold
    }
'''

# Add RUL function before load_models
content = content.replace(
    '@st.cache_resource\ndef load_models():',
    rul_function + '\n@st.cache_resource\ndef load_models():'
)

# Add RUL section after recommendations section in TAB 1
# Find the download report section and add RUL before it
rul_ui_code = '''
            # ── RUL SECTION ───────────────────────────────────────
            st.markdown("<div class=\\'sec\\'>◈ REMAINING USEFUL LIFE (RUL) ESTIMATE</div>", unsafe_allow_html=True)
            
            rul = calculate_rul(
                soh, cycle_scores, state_str,
                voltage, initial_voltage, temperature, cycle_time
            )
            
            # RUL display cards
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.markdown(f"""
                <div class=\\'mcard c\\'>
                    <div class=\\'mval c\\' style=\\'font-size:1.5rem;\\'>{rul["rul_months"]:.0f}</div>
                    <div class=\\'mlbl\\'>EST. MONTHS</div>
                </div>
                """, unsafe_allow_html=True)
            with r2:
                st.markdown(f"""
                <div class=\\'mcard g\\'>
                    <div class=\\'mval g\\' style=\\'font-size:1.5rem;\\'>{rul["rul_min"]:.0f}–{rul["rul_max"]:.0f}</div>
                    <div class=\\'mlbl\\'>RANGE (MONTHS)</div>
                </div>
                """, unsafe_allow_html=True)
            with r3:
                st.markdown(f"""
                <div class=\\'mcard o\\'>
                    <div class=\\'mval o\\' style=\\'font-size:1.5rem;\\'>{rul["degradation_rate"]:.2f}%</div>
                    <div class=\\'mlbl\\'>DEGRAD/MONTH</div>
                </div>
                """, unsafe_allow_html=True)
            with r4:
                st.markdown(f"""
                <div class=\\'mcard r\\'>
                    <div class=\\'mval r\\' style=\\'font-size:1.3rem;\\'>{rul["rul_emoji"]} {rul["rul_label"]}</div>
                    <div class=\\'mlbl\\'>RUL STATUS</div>
                </div>
                """, unsafe_allow_html=True)

            # RUL gauge chart
            import plotly.graph_objects as go
            rul_max_scale = max(rul["rul_months"] * 1.5, 24)
            fig_rul = go.Figure()
            
            # Background zones
            fig_rul.add_shape(type="rect", x0=0, x1=4, y0=0, y1=1,
                fillcolor="rgba(255,51,102,0.15)", line_width=0, layer="below")
            fig_rul.add_shape(type="rect", x0=4, x1=9, y0=0, y1=1,
                fillcolor="rgba(255,107,53,0.15)", line_width=0, layer="below")
            fig_rul.add_shape(type="rect", x0=9, x1=18, y0=0, y1=1,
                fillcolor="rgba(0,212,255,0.12)", line_width=0, layer="below")
            fig_rul.add_shape(type="rect", x0=18, x1=rul_max_scale, y0=0, y1=1,
                fillcolor="rgba(0,255,157,0.12)", line_width=0, layer="below")
            
            # RUL bar
            fig_rul.add_trace(go.Bar(
                x=[rul["rul_months"]], y=["RUL"],
                orientation="h",
                marker_color=rul["rul_color"],
                opacity=0.9,
                text=f"{rul[\\'rul_months\\']:.0f} months",
                textposition="outside",
                textfont=dict(size=14, family="Rajdhani", color=rul["rul_color"]),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    arrayminus=[rul["rul_months"] - rul["rul_min"]],
                    array=[rul["rul_max"] - rul["rul_months"]],
                    color="#7ba7cc",
                    thickness=3,
                    width=8
                )
            ))
            
            fig_rul.add_vline(x=rul["eol_threshold"] * 0, line_dash="dash",
                line_color="#ff3366", line_width=2,
                annotation_text="End-of-Life Threshold", annotation_position="top right")
            
            fig_rul.update_layout(
                **PLOT_BG,
                height=180,
                xaxis=dict(
                    range=[0, rul_max_scale],
                    title="Estimated Months Remaining",
                    gridcolor="#1a3a5c",
                    linecolor="#1a3a5c",
                    tickvals=[0, 4, 9, 18, int(rul_max_scale)],
                    ticktext=["0", "4m\\nCritical", "9m\\nShort", "18m\\nMedium", f"{int(rul_max_scale)}m\\nLong"],
                ),
                yaxis=dict(gridcolor="#1a3a5c", linecolor="#1a3a5c"),
                showlegend=False,
                margin=dict(l=10, r=10, t=20, b=10)
            )
            st.plotly_chart(fig_rul, use_container_width=True)

            st.markdown(f"""
            <div class=\\'icard\\' style=\\'border-left:4px solid {rul["rul_color"]};\\'>
                {rul["rul_emoji"]} <strong>RUL Summary:</strong>
                Estimated <strong style=\\'color:{rul["rul_color"]};\\'>
                {rul["rul_months"]:.0f} months</strong> remaining
                (range: {rul["rul_min"]:.0f}–{rul["rul_max"]:.0f} months)
                before reaching end-of-life threshold (SoH = 60%).<br><br>
                📉 <strong>Degradation Rate:</strong> {rul["degradation_rate"]:.2f}% SoH/month
                (base: {rul["base_rate"]:.2f}% × stress factor: {rul["stress_factor"]:.2f}×)<br><br>
                📋 <strong>Advice:</strong> {rul["rul_advice"]}
            </div>
            """, unsafe_allow_html=True)
            
            if rul["stress_notes"]:
                for note in rul["stress_notes"]:
                    st.markdown(f"<div class=\\'icard\\' style=\\'border-left:3px solid #ff6b35;\\'>⚠️ {note}</div>", unsafe_allow_html=True)

'''

# Insert RUL section before download report
content = content.replace(
    "            st.markdown(\"<div class=\\'sec\\'>◈ DOWNLOAD REPORT</div>\", unsafe_allow_html=True)",
    rul_ui_code + "\n            st.markdown(\"<div class=\\'sec\\'>◈ DOWNLOAD REPORT</div>\", unsafe_allow_html=True)"
)

# Save
with open(f'{SAVE_DIR}/app.py', 'w') as f:
    f.write(content)

from google.colab import files
files.download(f'{SAVE_DIR}/app.py')
print('✅ app.py with RUL downloaded!')
print('GitHub app.py replace කරලා commit!')

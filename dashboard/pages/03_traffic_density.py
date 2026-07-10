"""Traffic Density page - Density classification and heatmaps from real data."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

st.set_page_config(page_title="Traffic Density", layout="wide")
st.title("Traffic Density Analysis")

# ─── Density Level Indicator ─────────────────────────────────────────────────
st.markdown("### Current Traffic Density")

aggregator = st.session_state.get("aggregator")
final_counts = st.session_state.get("final_counts")
density_history = st.session_state.get("density_history")

if final_counts:
    total = final_counts.get("total", 0)

    # Determine density
    if total < 5:
        density, color = "Low", "#00FF00"
    elif total < 15:
        density, color = "Medium", "#FFD700"
    elif total < 30:
        density, color = "High", "#FF8C00"
    else:
        density, color = "Congested", "#FF0000"

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f'<div style="background-color:{color};padding:30px;border-radius:10px;text-align:center;">'
            f'<h1 style="color:white;margin:0;">{density}</h1>'
            f'<p style="color:white;margin:0;">{total} vehicles detected</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col2:
        # Density gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=total,
            title={"text": "Vehicles in Frame"},
            gauge={
                "axis": {"range": [0, 50]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 5], "color": "#E8F5E9"},
                    {"range": [5, 15], "color": "#FFF9C4"},
                    {"range": [15, 30], "color": "#FFE0B2"},
                    {"range": [30, 50], "color": "#FFCDD2"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 30,
                },
            },
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

else:
    st.info("Run detection first to see density analysis.")

st.markdown("---")

# ─── Density Over Time from Real Data ──────────────────────────────────────
st.markdown("### Density Over Time")

if density_history and len(density_history) > 0:
    # density_history is a list of DensityLevel enums from pipeline processing
    density_values = []
    density_labels = []
    for d in density_history:
        density_labels.append(d.value if hasattr(d, 'value') else str(d))
        mapping = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CONGESTED": 4}
        val = d.value if hasattr(d, 'value') else str(d)
        density_values.append(mapping.get(val.upper(), 1))

    density_df = pd.DataFrame({
        "Frame": range(len(density_values)),
        "Density Level": density_values,
        "Label": density_labels,
    })

    fig_density = px.line(
        density_df, x="Frame", y="Density Level",
        title="Traffic Density Level Over Frames",
        labels={"Density Level": "Density (1=Low, 2=Med, 3=High, 4=Congested)"},
    )
    fig_density.update_layout(height=350)
    fig_density.update_yaxes(tickvals=[1, 2, 3, 4], ticktext=["Low", "Medium", "High", "Congested"])
    st.plotly_chart(fig_density, use_container_width=True)

    # Density distribution pie chart
    from collections import Counter
    dist = Counter(density_labels)
    fig_dist = px.pie(
        names=list(dist.keys()),
        values=list(dist.values()),
        title="Density Level Distribution Across Video",
        color=list(dist.keys()),
        color_discrete_map={"LOW": "#00FF00", "MEDIUM": "#FFD700", "HIGH": "#FF8C00", "CONGESTED": "#FF0000"},
    )
    st.plotly_chart(fig_dist, use_container_width=True)

elif aggregator is not None:
    df = aggregator.get_dataframe()
    if not df.empty and "total" in df.columns:
        # Build density from aggregated counts
        def classify(count):
            if count < 5:
                return "Low"
            elif count < 15:
                return "Medium"
            elif count < 30:
                return "High"
            return "Congested"

        df["density"] = df["total"].apply(classify)
        fig = px.bar(
            df.reset_index(), x="timestamp", y="total",
            color="density",
            color_discrete_map={"Low": "#00FF00", "Medium": "#FFD700", "High": "#FF8C00", "Congested": "#FF0000"},
            title="Traffic Density Over Time (from Aggregated Counts)",
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run detection first to see density over time.")
else:
    st.info("Run detection on the **Live Detection** page to see density trends.")

st.markdown("---")

# ─── Density Classification Thresholds ───────────────────────────────────────
st.markdown("### Density Classification Thresholds")

threshold_col1, threshold_col2 = st.columns(2)
with threshold_col1:
    st.markdown("""
    | Level | Vehicle Count | Color |
    |-------|--------------|-------|
    | Low | 0 - 4 | Green |
    | Medium | 5 - 14 | Yellow |
    | High | 15 - 29 | Orange |
    | Congested | 30+ | Red |
    """)

with threshold_col2:
    st.markdown("""
    **Enhanced Classification** also considers vehicle speed:
    - Low speed (< 5 px/frame) + Medium count = **Congested**
    - Adaptive thresholds can be calibrated from historical data
    """)

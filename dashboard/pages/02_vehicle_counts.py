"""Vehicle Counts page - Time-series charts and summary statistics."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="Vehicle Counts", layout="wide")
st.title("Vehicle Count Analytics")

# Check if data exists from live detection
aggregator = st.session_state.get("aggregator")
final_counts = st.session_state.get("final_counts")

if aggregator is not None:
    df = aggregator.get_dataframe()

    if not df.empty:
        st.markdown("### Count Summary")

        # Metric cards
        if final_counts:
            cols = st.columns(min(len(final_counts), 6))
            for i, (cls, cnt) in enumerate(sorted(final_counts.items())):
                with cols[i % len(cols)]:
                    st.metric(cls.replace("_", " ").title(), cnt)

        st.markdown("---")

        # Time-series line chart
        st.markdown("### Vehicle Counts Over Time")
        fig_line = px.line(
            df.reset_index(),
            x="timestamp",
            y=[c for c in df.columns if c != "total"],
            title="Per-Class Vehicle Counts",
            labels={"value": "Count", "variable": "Class"},
        )
        fig_line.update_layout(height=400)
        st.plotly_chart(fig_line, use_container_width=True)

        # Total count line
        if "total" in df.columns:
            fig_total = px.line(
                df.reset_index(),
                x="timestamp",
                y="total",
                title="Total Vehicle Count Over Time",
            )
            fig_total.update_layout(height=300)
            st.plotly_chart(fig_total, use_container_width=True)

        st.markdown("---")

        # Class distribution pie chart
        st.markdown("### Vehicle Class Distribution")
        if final_counts:
            class_data = {k: v for k, v in final_counts.items() if k != "total" and v > 0}
            if class_data:
                fig_pie = px.pie(
                    names=list(class_data.keys()),
                    values=list(class_data.values()),
                    title="Distribution by Vehicle Class",
                )
                st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("---")

        # Raw data table
        st.markdown("### Raw Count Data")
        st.dataframe(df, use_container_width=True)

    else:
        st.warning("No count data available yet. Run detection first.")
else:
    st.info("Run detection on the **Live Detection** page first to see real count data here.")

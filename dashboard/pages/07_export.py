"""Export page - Download reports in CSV, JSON, or PDF format."""

import io
import json
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Export Data", layout="wide")
st.title("Export Traffic Data")

st.markdown("Download vehicle count data and reports in various formats.")

# ─── Data Source ──────────────────────────────────────────────────────────────
aggregator = st.session_state.get("aggregator")
final_counts = st.session_state.get("final_counts")

has_data = aggregator is not None and final_counts is not None

if not has_data:
    st.warning("No detection data available. Run detection on the **Live Detection** page first.")
    st.markdown("---")
    st.markdown("### Demo Export (Sample Data)")

    # Generate sample data for demo
    np.random.seed(42)
    timestamps = pd.date_range("2025-01-01 06:00", periods=48, freq="30min")
    demo_df = pd.DataFrame({
        "timestamp": timestamps,
        "car": np.random.poisson(15, 48),
        "bus": np.random.poisson(3, 48),
        "truck": np.random.poisson(5, 48),
        "auto": np.random.poisson(8, 48),
        "two_wheeler": np.random.poisson(12, 48),
    })
    demo_df["total"] = demo_df[["car", "bus", "truck", "auto", "two_wheeler"]].sum(axis=1)
    final_counts = {
        "car": int(demo_df["car"].sum()),
        "bus": int(demo_df["bus"].sum()),
        "truck": int(demo_df["truck"].sum()),
        "auto": int(demo_df["auto"].sum()),
        "two_wheeler": int(demo_df["two_wheeler"].sum()),
        "total": int(demo_df["total"].sum()),
    }
    df = demo_df
else:
    df = aggregator.get_dataframe()

st.markdown("---")

# ─── Export Options ──────────────────────────────────────────────────────────
st.markdown("### Choose Export Format")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### CSV Export")
    st.markdown("Time-series count data in CSV format.")

    csv_buffer = io.StringIO()
    df_export = df.reset_index() if hasattr(df, 'index') and df.index.name else df
    df_export.to_csv(csv_buffer, index=False)

    st.download_button(
        label="Download CSV",
        data=csv_buffer.getvalue(),
        file_name=f"vehicle_counts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        type="primary",
    )

with col2:
    st.markdown("#### JSON Export")
    st.markdown("Summary report in JSON format.")

    json_data = {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_vehicles": final_counts.get("total", 0),
            "per_class": {k: v for k, v in final_counts.items() if k != "total"},
        },
    }
    json_str = json.dumps(json_data, indent=2)

    st.download_button(
        label="Download JSON",
        data=json_str,
        file_name=f"vehicle_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        type="primary",
    )

with col3:
    st.markdown("#### Excel Export")
    st.markdown("Spreadsheet with count data.")

    excel_buffer = io.BytesIO()
    df_export = df.reset_index() if hasattr(df, 'index') and df.index.name else df
    df_export.to_excel(excel_buffer, index=False, engine="openpyxl")

    st.download_button(
        label="Download Excel",
        data=excel_buffer.getvalue(),
        file_name=f"vehicle_counts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )

st.markdown("---")

# ─── Data Preview ────────────────────────────────────────────────────────────
st.markdown("### Data Preview")
st.dataframe(df, use_container_width=True)

# ─── Summary Stats ───────────────────────────────────────────────────────────
st.markdown("### Summary Statistics")
if final_counts:
    for cls, count in sorted(final_counts.items()):
        st.markdown(f"- **{cls.replace('_', ' ').title()}**: {count}")

"""Dashboard page for Traffic Map Visualization & Route Suggestions - Real data."""

import streamlit as st
import streamlit.components.v1 as components
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Traffic Map", page_icon=":world_map:", layout="wide")
st.title(":world_map: Live Traffic Map")
st.markdown("Real-time traffic density on map with route suggestions.")
st.markdown("---")

# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.header("Map Settings")
show_heatmap = st.sidebar.checkbox("Show Traffic Heatmap", value=True)
show_cameras = st.sidebar.checkbox("Show Camera Markers", value=True)
map_style = st.sidebar.selectbox("Map Style", ["OpenStreetMap", "Light Mode", "Dark Mode"])

st.sidebar.markdown("---")
st.sidebar.header("Route Settings")
show_routes = st.sidebar.checkbox("Enable Route Suggestions", value=True)

# ─── Main Content ────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Traffic Map", "Route Planner", "Camera Management"])

with tab1:
    st.subheader("Live Traffic Density Map")

    try:
        from src.maps.traffic_map import TrafficMapVisualizer, CameraLocation

        map_viz = TrafficMapVisualizer()

        # Load real data from session if available
        final_counts = st.session_state.get("final_counts", {})

        map_viz.load_demo_cameras()

        # Update first camera with real detection data
        if final_counts:
            total = final_counts.get("total", 0)
            cameras = map_viz.get_cameras()
            if cameras:
                cameras[0].vehicle_count = total
                if total < 5:
                    cameras[0].density_level = "low"
                elif total < 15:
                    cameras[0].density_level = "medium"
                elif total < 30:
                    cameras[0].density_level = "high"
                else:
                    cameras[0].density_level = "congested"

        # User-added cameras from session
        user_cameras = st.session_state.get("user_cameras", [])
        for uc in user_cameras:
            map_viz.add_camera(CameraLocation(**uc))

        traffic_map = map_viz.create_map(show_heatmap=show_heatmap)
        map_html = traffic_map._repr_html_()
        components.html(map_html, height=600, scrolling=True)

        st.markdown("---")
        st.markdown("### Camera Status")

        cameras = map_viz.get_cameras()
        if cameras:
            cols = st.columns(min(len(cameras), 4))
            for i, cam in enumerate(cameras):
                with cols[i % 4]:
                    color_emoji = {
                        "low": "🟢", "medium": "🟡",
                        "high": "🟠", "congested": "🔴",
                    }.get(cam.density_level, "⚪")
                    st.markdown(f"{color_emoji} **{cam.name}**")
                    st.caption(f"{cam.vehicle_count} vehicles -- {cam.density_level.upper()}")

    except ImportError as e:
        st.error(f"Missing dependency: {e}")
        st.info("Install with: `pip install folium`")

with tab2:
    st.subheader("Route Planner with Traffic Overlay")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Start Point**")
        start_lat = st.number_input("Start Latitude", value=13.3411, format="%.4f", key="start_lat")
        start_lon = st.number_input("Start Longitude", value=77.1010, format="%.4f", key="start_lon")

    with col2:
        st.markdown("**Destination**")
        end_lat = st.number_input("End Latitude", value=13.3290, format="%.4f", key="end_lat")
        end_lon = st.number_input("End Longitude", value=77.1100, format="%.4f", key="end_lon")

    st.markdown("**Quick Presets (Tumkur)**")
    preset_col1, preset_col2, preset_col3 = st.columns(3)
    with preset_col1:
        if st.button("MG Road -> Bus Stand"):
            start_lat, start_lon = 13.3411, 77.1010
            end_lat, end_lon = 13.3379, 77.1173
    with preset_col2:
        if st.button("SSI Circle -> Railway Station"):
            start_lat, start_lon = 13.3325, 77.1050
            end_lat, end_lon = 13.3340, 77.1250
    with preset_col3:
        if st.button("NH-48 -> Siddaganga"):
            start_lat, start_lon = 13.3500, 77.1300
            end_lat, end_lon = 13.3150, 77.0980

    if st.button("Find Routes", type="primary"):
        try:
            from src.maps.route_suggester import RouteSuggester

            rs = RouteSuggester()
            routes = rs.get_routes((start_lat, start_lon), (end_lat, end_lon))

            route_map = rs.get_route_map(
                (start_lat, start_lon), (end_lat, end_lon), routes
            )
            components.html(route_map._repr_html_(), height=500, scrolling=True)

            st.markdown("### Route Options")
            for route in routes:
                rec = " :star: **RECOMMENDED**" if route.is_recommended else ""
                traffic_emoji = {
                    "low": "🟢", "medium": "🟡",
                    "high": "🟠", "congested": "🔴",
                }.get(route.traffic_level, "⚪")

                with st.expander(f"Route {route.route_id}: {route.summary}{rec}"):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Distance", f"{route.distance_km} km")
                    with c2:
                        st.metric("Duration", f"{route.duration_min} min")
                    with c3:
                        st.markdown(f"{traffic_emoji} **Traffic: {route.traffic_level.upper()}**")

            if rs._api_key:
                st.success("Routes from OpenRouteService API (real data)")
            else:
                st.info("Using demo routes. Set `ORS_API_KEY` in `.env` for real routing data.")

        except ImportError as e:
            st.error(f"Missing dependency: {e}")

with tab3:
    st.subheader("Camera Location Management")
    st.markdown("Add camera GPS locations for map display.")

    with st.form("add_camera"):
        c1, c2, c3 = st.columns(3)
        with c1:
            cam_id = st.text_input("Camera ID", placeholder="cam_09")
            cam_name = st.text_input("Camera Name", placeholder="New Junction")
        with c2:
            cam_lat = st.number_input("Latitude", value=13.3400, format="%.4f")
            cam_lon = st.number_input("Longitude", value=77.1100, format="%.4f")
        with c3:
            cam_density = st.selectbox("Initial Density", ["low", "medium", "high", "congested"])
            cam_count = st.number_input("Vehicle Count", value=0, min_value=0)

        if st.form_submit_button("Add Camera to Map"):
            if "user_cameras" not in st.session_state:
                st.session_state["user_cameras"] = []
            st.session_state["user_cameras"].append({
                "camera_id": cam_id or f"cam_{len(st.session_state['user_cameras']) + 10}",
                "name": cam_name or "New Camera",
                "latitude": cam_lat,
                "longitude": cam_lon,
                "density_level": cam_density,
                "vehicle_count": cam_count,
            })
            st.success(f"Camera '{cam_name}' added! Refresh the Traffic Map tab to see it.")
            st.rerun()

    # Show existing user cameras
    user_cameras = st.session_state.get("user_cameras", [])
    if user_cameras:
        import pandas as pd
        st.markdown("### Added Cameras")
        st.dataframe(pd.DataFrame(user_cameras), use_container_width=True)
        if st.button("Clear All Added Cameras"):
            st.session_state["user_cameras"] = []
            st.rerun()

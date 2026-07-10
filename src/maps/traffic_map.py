"""Traffic map visualization using Folium (OpenStreetMap)."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import folium
from folium.plugins import HeatMap, MarkerCluster


@dataclass
class CameraLocation:
    """A camera's geographic location with traffic data."""
    camera_id: str
    name: str
    latitude: float
    longitude: float
    density_level: str = "low"  # "low", "medium", "high", "congested"
    vehicle_count: int = 0
    is_active: bool = True


DENSITY_COLORS = {
    "low": "green",
    "medium": "orange",
    "high": "red",
    "congested": "darkred",
}

DENSITY_ICONS = {
    "low": "ok-sign",
    "medium": "info-sign",
    "high": "warning-sign",
    "congested": "remove-sign",
}


class TrafficMapVisualizer:
    """Creates interactive traffic maps with camera locations and density overlays.

    Uses Folium (OpenStreetMap) — completely free, no API key needed.
    """

    def __init__(
        self,
        center_lat: float = 13.3379,
        center_lon: float = 77.1173,
        zoom: int = 14,
        city_name: str = "Tumkur, Karnataka",
    ):
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.zoom = zoom
        self.city_name = city_name
        self._cameras: Dict[str, CameraLocation] = {}

    def add_camera(self, camera: CameraLocation):
        """Register a camera location on the map."""
        self._cameras[camera.camera_id] = camera

    def update_camera_density(self, camera_id: str, density: str, count: int):
        """Update traffic density for a camera."""
        if camera_id in self._cameras:
            self._cameras[camera_id].density_level = density
            self._cameras[camera_id].vehicle_count = count

    def create_map(self, show_heatmap: bool = True) -> folium.Map:
        """Generate an interactive Folium map with traffic overlays.

        Args:
            show_heatmap: Whether to add a vehicle density heatmap layer.

        Returns:
            folium.Map object (render with ._repr_html_() or .save()).
        """
        m = folium.Map(
            location=[self.center_lat, self.center_lon],
            zoom_start=self.zoom,
            tiles="OpenStreetMap",
        )

        # Add tile layers
        folium.TileLayer("cartodbpositron", name="Light Mode").add_to(m)
        folium.TileLayer("cartodbdark_matter", name="Dark Mode").add_to(m)

        # Camera markers
        marker_cluster = MarkerCluster(name="Cameras").add_to(m)

        heatmap_data = []

        for cam in self._cameras.values():
            color = DENSITY_COLORS.get(cam.density_level, "gray")
            icon_name = DENSITY_ICONS.get(cam.density_level, "info-sign")

            popup_html = f"""
            <div style="width:200px">
                <h4>{cam.name}</h4>
                <b>Status:</b> {"Active" if cam.is_active else "Offline"}<br>
                <b>Density:</b> <span style="color:{color}">{cam.density_level.upper()}</span><br>
                <b>Vehicle Count:</b> {cam.vehicle_count}<br>
                <b>Camera ID:</b> {cam.camera_id}
            </div>
            """

            folium.Marker(
                location=[cam.latitude, cam.longitude],
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"{cam.name} — {cam.density_level.upper()} ({cam.vehicle_count} vehicles)",
                icon=folium.Icon(color=color, icon=icon_name, prefix="glyphicon"),
            ).add_to(marker_cluster)

            # Density circle
            folium.CircleMarker(
                location=[cam.latitude, cam.longitude],
                radius=max(10, cam.vehicle_count),
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.4,
                weight=2,
            ).add_to(m)

            # Heatmap data point (weighted by vehicle count)
            if cam.vehicle_count > 0:
                heatmap_data.append([cam.latitude, cam.longitude, cam.vehicle_count])

        # Add heatmap layer
        if show_heatmap and heatmap_data:
            HeatMap(
                heatmap_data,
                name="Traffic Heatmap",
                min_opacity=0.3,
                radius=30,
                blur=20,
                gradient={0.2: "blue", 0.4: "lime", 0.6: "yellow", 0.8: "orange", 1.0: "red"},
            ).add_to(m)

        # Layer control
        folium.LayerControl().add_to(m)

        return m

    def save_map(self, filepath: str = "outputs/traffic_map.html"):
        """Save the map to an HTML file."""
        m = self.create_map()
        m.save(filepath)
        return filepath

    def get_map_html(self) -> str:
        """Get map HTML string for embedding in Streamlit."""
        m = self.create_map()
        return m._repr_html_()

    def get_cameras(self) -> List[CameraLocation]:
        return list(self._cameras.values())

    def load_demo_cameras(self):
        """Load demo camera locations for Tumkur city."""
        demo_cameras = [
            CameraLocation("cam_01", "MG Road Junction", 13.3411, 77.1010, "high", 28),
            CameraLocation("cam_02", "Bus Stand Circle", 13.3379, 77.1173, "congested", 45),
            CameraLocation("cam_03", "SSI Circle", 13.3325, 77.1050, "medium", 12),
            CameraLocation("cam_04", "Kyatsandra Gate", 13.3450, 77.1200, "low", 4),
            CameraLocation("cam_05", "NH-48 Bypass", 13.3500, 77.1300, "high", 32),
            CameraLocation("cam_06", "BH Road", 13.3290, 77.1100, "medium", 18),
            CameraLocation("cam_07", "Siddaganga Entrance", 13.3150, 77.0980, "low", 6),
            CameraLocation("cam_08", "Railway Station", 13.3340, 77.1250, "high", 25),
        ]
        for cam in demo_cameras:
            self.add_camera(cam)

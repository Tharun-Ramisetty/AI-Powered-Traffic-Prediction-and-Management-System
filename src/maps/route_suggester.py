"""Route suggestion using OpenRouteService API (free, open-source)."""

import os
import requests
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import folium
from loguru import logger

from src.utils.retry import with_retry

_TRANSIENT_HTTP = (requests.ConnectionError, requests.Timeout)


@dataclass
class RouteOption:
    """A suggested route between two points."""
    route_id: int
    distance_km: float
    duration_min: float
    geometry: List[List[float]]  # list of [lon, lat] coordinates
    traffic_level: str = "unknown"  # estimated from camera data
    summary: str = ""
    is_recommended: bool = False


class RouteSuggester:
    """Suggests routes between locations and overlays traffic data.

    Uses OpenRouteService API (free tier: 2000 requests/day).
    Sign up at: https://openrouteservice.org/dev/#/signup
    """

    ORS_BASE_URL = "https://api.openrouteservice.org"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ORS_API_KEY", "")
        self._route_cache: Dict[str, List[RouteOption]] = {}

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def get_routes(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        profile: str = "driving-car",
        alternatives: int = 3,
    ) -> List[RouteOption]:
        """Get route options between two points.

        Args:
            start: (latitude, longitude) of start point.
            end: (latitude, longitude) of end point.
            profile: "driving-car", "driving-hgv", "cycling-regular", "foot-walking"
            alternatives: Number of alternative routes to request.

        Returns:
            List of RouteOption objects sorted by duration.
        """
        if not self.is_configured:
            return self._get_demo_routes(start, end)

        if not (-90 <= start[0] <= 90 and -180 <= start[1] <= 180):
            raise ValueError(f"start coordinates out of range: {start}")
        if not (-90 <= end[0] <= 90 and -180 <= end[1] <= 180):
            raise ValueError(f"end coordinates out of range: {end}")
        if not 1 <= alternatives <= 5:
            raise ValueError(
                f"alternatives must be 1..5 (ORS limit), got {alternatives}"
            )

        coordinates = [[start[1], start[0]], [end[1], end[0]]]

        try:
            data = self._request_routes(profile, coordinates, alternatives)
        except _TRANSIENT_HTTP as exc:
            logger.warning("ORS transient failure ({}): falling back to demo "
                           "routes — caller should retry later", type(exc).__name__)
            return self._get_demo_routes(start, end)
        except requests.HTTPError as exc:
            logger.error("ORS HTTP error: {}. Using demo routes.", exc)
            return self._get_demo_routes(start, end)

        routes = []
        for i, feature in enumerate(data.get("features", [])):
            props = feature.get("properties", {})
            summary = props.get("summary", {})
            geometry = feature.get("geometry", {}).get("coordinates", [])

            route = RouteOption(
                route_id=i + 1,
                distance_km=round(summary.get("distance", 0) / 1000, 2),
                duration_min=round(summary.get("duration", 0) / 60, 1),
                geometry=geometry,
                summary=f"Route {i + 1}",
                is_recommended=(i == 0),
            )
            routes.append(route)

        return sorted(routes, key=lambda r: r.duration_min)

    @with_retry(exceptions=_TRANSIENT_HTTP, max_attempts=3, initial_wait=1.0)
    def _request_routes(
        self, profile: str, coordinates: list, alternatives: int
    ) -> Dict:
        response = requests.post(
            f"{self.ORS_BASE_URL}/v2/directions/{profile}/geojson",
            json={
                "coordinates": coordinates,
                "alternative_routes": {
                    "target_count": alternatives,
                    "share_factor": 0.6,
                    "weight_factor": 1.6,
                },
                "geometry": True,
                "instructions": False,
            },
            headers={
                "Authorization": self.api_key,
                "Content-Type": "application/json",
            },
            timeout=10,
        )
        response.raise_for_status()
        return response.json()

    def get_route_map(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        routes: Optional[List[RouteOption]] = None,
        camera_data: Optional[List[Dict]] = None,
    ) -> folium.Map:
        """Create a map showing route options with traffic overlay.

        Args:
            start: Start point (lat, lon).
            end: End point (lat, lon).
            routes: Route options (fetched if None).
            camera_data: Optional camera density data for overlay.

        Returns:
            Folium map with routes drawn.
        """
        if routes is None:
            routes = self.get_routes(start, end)

        center_lat = (start[0] + end[0]) / 2
        center_lon = (start[1] + end[1]) / 2

        m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

        route_colors = ["blue", "green", "orange", "purple", "gray"]

        for i, route in enumerate(routes):
            color = route_colors[i % len(route_colors)]
            if route.is_recommended:
                color = "blue"
                weight = 6
            else:
                weight = 4

            # Convert [lon, lat] to [lat, lon] for folium
            latlngs = [[coord[1], coord[0]] for coord in route.geometry]

            tooltip = f"Route {route.route_id}: {route.distance_km} km, {route.duration_min} min"
            if route.traffic_level != "unknown":
                tooltip += f" ({route.traffic_level} traffic)"

            folium.PolyLine(
                locations=latlngs,
                color=color,
                weight=weight,
                opacity=0.8,
                tooltip=tooltip,
                popup=f"<b>Route {route.route_id}</b><br>"
                       f"Distance: {route.distance_km} km<br>"
                       f"Duration: {route.duration_min} min<br>"
                       f"Traffic: {route.traffic_level}",
            ).add_to(m)

        # Start marker
        folium.Marker(
            location=list(start),
            popup="Start",
            icon=folium.Icon(color="green", icon="play", prefix="glyphicon"),
        ).add_to(m)

        # End marker
        folium.Marker(
            location=list(end),
            popup="Destination",
            icon=folium.Icon(color="red", icon="stop", prefix="glyphicon"),
        ).add_to(m)

        # Camera overlays
        if camera_data:
            for cam in camera_data:
                density_color = {
                    "low": "green", "medium": "orange",
                    "high": "red", "congested": "darkred",
                }.get(cam.get("density", ""), "gray")

                folium.CircleMarker(
                    location=[cam["lat"], cam["lon"]],
                    radius=8,
                    color=density_color,
                    fill=True,
                    fill_opacity=0.6,
                    tooltip=f"{cam.get('name', '')} — {cam.get('density', '').upper()}",
                ).add_to(m)

        return m

    def estimate_route_traffic(
        self, route: RouteOption, camera_locations: List[Dict]
    ) -> str:
        """Estimate traffic level on a route based on nearby camera data.

        Checks which cameras are close to the route path and averages their density.
        """
        if not route.geometry or not camera_locations:
            return "unknown"

        import math

        density_scores = {"low": 1, "medium": 2, "high": 3, "congested": 4}
        nearby_scores = []

        for coord in route.geometry[::5]:  # Sample every 5th point
            route_lat, route_lon = coord[1], coord[0]

            for cam in camera_locations:
                dist = math.sqrt(
                    (route_lat - cam["lat"]) ** 2 + (route_lon - cam["lon"]) ** 2
                )
                if dist < 0.005:  # ~500m radius
                    score = density_scores.get(cam.get("density", "low"), 1)
                    nearby_scores.append(score)

        if not nearby_scores:
            return "unknown"

        avg = sum(nearby_scores) / len(nearby_scores)
        if avg <= 1.5:
            return "low"
        elif avg <= 2.5:
            return "medium"
        elif avg <= 3.5:
            return "high"
        return "congested"

    def _get_demo_routes(
        self, start: Tuple[float, float], end: Tuple[float, float]
    ) -> List[RouteOption]:
        """Generate demo routes when API key is not available."""
        import math

        dist = math.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2) * 111

        # Straight-line route
        route1_geom = [
            [start[1], start[0]],
            [(start[1] + end[1]) / 2, (start[0] + end[0]) / 2],
            [end[1], end[0]],
        ]

        # Curved alternative 1
        mid_lat = (start[0] + end[0]) / 2 + 0.005
        mid_lon = (start[1] + end[1]) / 2 + 0.003
        route2_geom = [
            [start[1], start[0]],
            [mid_lon, mid_lat],
            [end[1], end[0]],
        ]

        # Curved alternative 2
        mid_lat2 = (start[0] + end[0]) / 2 - 0.004
        mid_lon2 = (start[1] + end[1]) / 2 - 0.005
        route3_geom = [
            [start[1], start[0]],
            [mid_lon2, mid_lat2],
            [end[1], end[0]],
        ]

        return [
            RouteOption(1, round(dist, 2), round(dist * 3, 1), route1_geom,
                       "medium", "Via Main Road", True),
            RouteOption(2, round(dist * 1.2, 2), round(dist * 2.5, 1), route2_geom,
                       "low", "Via Bypass Road", False),
            RouteOption(3, round(dist * 1.5, 2), round(dist * 4, 1), route3_geom,
                       "high", "Via City Center", False),
        ]

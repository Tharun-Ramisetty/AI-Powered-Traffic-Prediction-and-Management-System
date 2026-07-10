"""Global configuration for the AI-Powered Vehicle Count Prediction system."""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CONFIG_DIR = PROJECT_ROOT / "config"

# ─── Vehicle Classes ─────────────────────────────────────────────────────────
VEHICLE_CLASSES = ["Auto", "Bike", "Bus", "Car", "Scooty", "Taxi", "Tempo", "Toto", "Truck"]
NUM_CLASSES = len(VEHICLE_CLASSES)

CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Auto": (0, 255, 255),       # Yellow
    "Bike": (0, 255, 50),        # Bright Green
    "Bus": (255, 0, 0),          # Blue
    "Car": (50, 255, 50),        # Bright Lime Green
    "Scooty": (128, 0, 128),     # Purple
    "Taxi": (0, 255, 128),       # Spring Green
    "Tempo": (0, 165, 255),      # Orange
    "Toto": (255, 128, 0),       # Cyan-ish
    "Truck": (0, 0, 255),        # Red
}


# ─── Detection Config ────────────────────────────────────────────────────────
@dataclass
class DetectionConfig:
    model_name: str = "yolov8"
    model_path: str = str(MODELS_DIR / "yolov8" / "best.pt")
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    img_size: int = 640
    device: str = "auto"  # "cpu", "cuda:0", or "auto"


# ─── Tracking Config ─────────────────────────────────────────────────────────
@dataclass
class TrackingConfig:
    tracker_type: str = "bytetrack"  # "deepsort" or "bytetrack"
    max_age: int = 30                # Max frames to keep lost track
    min_hits: int = 3                # Min detections before track is confirmed
    iou_threshold: float = 0.3


# ─── Counting Config ─────────────────────────────────────────────────────────
@dataclass
class CountingConfig:
    mode: str = "line"                  # "line" or "zone"
    line_y_fraction: float = 0.6        # Fraction of frame height for counting line
    direction: str = "both"             # "up", "down", "both"
    zone_points: List[Tuple[int, int]] = field(default_factory=list)
    aggregation_window_seconds: int = 300  # 5-minute windows


# ─── Density Classification Config ───────────────────────────────────────────
@dataclass
class DensityConfig:
    low_threshold: int = 5
    medium_threshold: int = 15
    high_threshold: int = 30
    # Above high_threshold = Congested
    use_adaptive: bool = False


# ─── LSTM Prediction Config ──────────────────────────────────────────────────
@dataclass
class LSTMConfig:
    sequence_length: int = 24       # Hours of history as input
    prediction_horizon: int = 6     # Hours ahead to predict
    hidden_size: int = 128
    num_layers: int = 2
    num_heads: int = 4              # Attention heads
    dropout: float = 0.2
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    include_weather: bool = True
    model_path: str = str(MODELS_DIR / "lstm" / "traffic_predictor.pth")


# ─── Weather Config ──────────────────────────────────────────────────────────
@dataclass
class WeatherConfig:
    api_key_env: str = "OPENWEATHER_API_KEY"
    city: str = "Tumkur,IN"
    features: List[str] = field(default_factory=lambda: [
        "temperature", "humidity", "rain_1h", "visibility", "wind_speed"
    ])


# ─── Accident Detection Config ──────────────────────────────────────────────
@dataclass
class AccidentDetectionConfig:
    deceleration_threshold: float = 15.0
    collision_iou_threshold: float = 0.3
    stationary_frames: int = 90
    min_speed_for_sudden_stop: float = 8.0
    cooldown_seconds: float = 10.0


# ─── ANPR Config ────────────────────────────────────────────────────────────
@dataclass
class ANPRConfig:
    plate_detector_mode: str = "cascade"  # "cascade" or "yolo"
    plate_model_path: str = str(MODELS_DIR / "anpr" / "plate_detector.pt")
    ocr_languages: List[str] = field(default_factory=lambda: ["en"])
    ocr_gpu: bool = False
    confidence_threshold: float = 0.5
    flagged_vehicles_db: str = str(DATA_DIR / "flagged_vehicles.json")


# ─── Emergency Vehicle Detection Config ─────────────────────────────────────
@dataclass
class EmergencyDetectionConfig:
    color_threshold: float = 0.15
    min_vehicle_area: int = 5000
    use_ocr: bool = True
    ocr_languages: List[str] = field(default_factory=lambda: ["en"])
    green_duration: float = 30.0
    yellow_duration: float = 3.0
    signal_cooldown: float = 15.0


# ─── Alert System Config ───────────────────────────────────────────────────
@dataclass
class AlertConfig:
    sms_enabled: bool = False
    notification_enabled: bool = True
    min_priority: str = "MEDIUM"  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    cooldown_seconds: float = 30.0
    notification_storage: str = str(OUTPUTS_DIR / "notifications.json")
    sms_recipients: List[str] = field(default_factory=list)


# ─── Maps & Route Config ───────────────────────────────────────────────────
@dataclass
class MapsConfig:
    center_lat: float = 13.3379
    center_lon: float = 77.1173
    zoom: int = 14
    city_name: str = "Tumkur, Karnataka"
    ors_api_key_env: str = "ORS_API_KEY"


# ─── Firebase Config ─────────────────────────────────────────────────────
@dataclass
class FirebaseConfig:
    credentials_path_env: str = "FIREBASE_CREDENTIALS_PATH"
    project_id_env: str = "FIREBASE_PROJECT_ID"
    device_tokens_file: str = str(DATA_DIR / "fcm_device_tokens.json")


# ─── Pipeline Config (combines all) ─────────────────────────────────────────
@dataclass
class PipelineConfig:
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    counting: CountingConfig = field(default_factory=CountingConfig)
    density: DensityConfig = field(default_factory=DensityConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    accident: AccidentDetectionConfig = field(default_factory=AccidentDetectionConfig)
    anpr: ANPRConfig = field(default_factory=ANPRConfig)
    emergency: EmergencyDetectionConfig = field(default_factory=EmergencyDetectionConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    maps: MapsConfig = field(default_factory=MapsConfig)
    firebase: FirebaseConfig = field(default_factory=FirebaseConfig)
    output_video: bool = False
    output_dir: str = str(OUTPUTS_DIR)

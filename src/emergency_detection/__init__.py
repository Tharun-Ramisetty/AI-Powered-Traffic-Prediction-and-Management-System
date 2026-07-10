"""Emergency vehicle detection module - ambulance, fire truck, police."""

from .emergency_detector import EmergencyVehicleDetector, EmergencyEvent
from .signal_controller import SignalController

__all__ = ["EmergencyVehicleDetector", "EmergencyEvent", "SignalController"]

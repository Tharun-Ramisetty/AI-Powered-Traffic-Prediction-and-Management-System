"""Traffic signal controller for emergency vehicle priority."""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

from .emergency_detector import EmergencyEvent


class SignalState(Enum):
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"


@dataclass
class TrafficSignal:
    """Represents a traffic signal at an intersection."""
    signal_id: str
    direction: str  # "north", "south", "east", "west"
    state: SignalState = SignalState.RED
    is_emergency_override: bool = False
    override_until: float = 0.0  # timestamp when override expires


@dataclass
class SignalChangeLog:
    """Log entry for signal state changes."""
    timestamp: float
    signal_id: str
    old_state: str
    new_state: str
    reason: str
    emergency_event_id: str = ""


class SignalController:
    """Controls traffic signals based on emergency vehicle detection.

    When an emergency vehicle is detected approaching an intersection:
    1. The signal in the emergency vehicle's direction turns GREEN
    2. All other signals turn RED
    3. After the vehicle passes or a timeout, signals return to normal

    In production, this would interface with actual traffic signal hardware
    via IoT protocols (MQTT, REST API, etc.).
    """

    def __init__(
        self,
        green_duration: float = 30.0,
        yellow_duration: float = 3.0,
        cooldown_seconds: float = 15.0,
    ):
        self.green_duration = green_duration
        self.yellow_duration = yellow_duration
        self.cooldown_seconds = cooldown_seconds

        self._signals: Dict[str, TrafficSignal] = {}
        self._change_log: List[SignalChangeLog] = []
        self._last_override_time: float = 0.0
        self._normal_cycle: Dict[str, SignalState] = {}

        # Initialize default 4-way intersection
        for direction in ["north", "south", "east", "west"]:
            sig_id = f"signal_{direction}"
            self._signals[sig_id] = TrafficSignal(
                signal_id=sig_id,
                direction=direction,
                state=SignalState.RED,
            )

    def handle_emergency(self, event: EmergencyEvent, approach_direction: str = "north"):
        """Override signals for an approaching emergency vehicle.

        Args:
            event: The detected emergency vehicle event.
            approach_direction: Direction the emergency vehicle is approaching from.
        """
        current_time = time.time()

        # Cooldown check
        if current_time - self._last_override_time < self.cooldown_seconds:
            return

        # Save current normal state
        for sig_id, signal in self._signals.items():
            if not signal.is_emergency_override:
                self._normal_cycle[sig_id] = signal.state

        # Set approaching direction to GREEN, all others to RED
        for sig_id, signal in self._signals.items():
            old_state = signal.state.value
            if signal.direction == approach_direction:
                signal.state = SignalState.GREEN
                signal.is_emergency_override = True
                signal.override_until = current_time + self.green_duration
            else:
                signal.state = SignalState.RED
                signal.is_emergency_override = True
                signal.override_until = current_time + self.green_duration

            self._change_log.append(SignalChangeLog(
                timestamp=current_time,
                signal_id=sig_id,
                old_state=old_state,
                new_state=signal.state.value,
                reason=f"Emergency: {event.vehicle_type} approaching from {approach_direction}",
                emergency_event_id=event.event_id,
            ))

        self._last_override_time = current_time

    def update(self):
        """Check and revert expired emergency overrides."""
        current_time = time.time()

        for sig_id, signal in self._signals.items():
            if signal.is_emergency_override and current_time >= signal.override_until:
                old_state = signal.state.value
                # Revert to normal cycle state or RED
                signal.state = self._normal_cycle.get(sig_id, SignalState.RED)
                signal.is_emergency_override = False
                signal.override_until = 0.0

                self._change_log.append(SignalChangeLog(
                    timestamp=current_time,
                    signal_id=sig_id,
                    old_state=old_state,
                    new_state=signal.state.value,
                    reason="Emergency override expired, reverting to normal",
                ))

    def get_signal_states(self) -> Dict[str, Dict]:
        """Get current state of all signals."""
        return {
            sig_id: {
                "direction": s.direction,
                "state": s.state.value,
                "is_emergency": s.is_emergency_override,
                "override_remaining": max(0, s.override_until - time.time())
                if s.is_emergency_override else 0,
            }
            for sig_id, s in self._signals.items()
        }

    def get_change_log(self, last_n: int = 20) -> List[SignalChangeLog]:
        return self._change_log[-last_n:]

    @property
    def is_in_emergency_mode(self) -> bool:
        return any(s.is_emergency_override for s in self._signals.values())

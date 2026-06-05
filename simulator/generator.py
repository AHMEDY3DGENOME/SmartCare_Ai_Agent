import random
from datetime import datetime

from simulator.wifi_csi.csi_generator import generate_csi_waveform


CURRENT_SCENARIO = {
    "mode": "auto"
}


def set_simulation_mode(mode: str) -> dict:
    """
    Set the current simulation mode.
    Available modes:
    auto, normal, warning, critical, fall, no_patient
    """

    allowed_modes = [
        "auto",
        "normal",
        "warning",
        "critical",
        "fall",
        "no_patient",
    ]

    if mode not in allowed_modes:
        return {
            "success": False,
            "message": f"Invalid mode '{mode}'. Allowed modes: {allowed_modes}",
            "current_mode": CURRENT_SCENARIO["mode"],
        }

    CURRENT_SCENARIO["mode"] = mode

    return {
        "success": True,
        "message": f"Simulation mode changed to {mode}",
        "current_mode": CURRENT_SCENARIO["mode"],
    }


def get_simulation_mode() -> str:
    """
    Return the current simulation mode.
    """

    return CURRENT_SCENARIO["mode"]


def _build_sample(
    mode: str,
    presence: bool,
    movement_level: float,
    breathing_rate: int,
    heart_rate: int,
    signal_strength: float,
    scenario: str,
) -> dict:
    """
    Build the final simulated sample including WiFi CSI waveform.
    """

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "presence": presence,
        "movement_level": movement_level,
        "breathing_rate": breathing_rate,
        "heart_rate": heart_rate,
        "signal_strength": signal_strength,
        "scenario": scenario,
        "mode": mode,
        "wifi_csi": generate_csi_waveform(mode),
    }


def generate_wifi_csi_sample() -> dict:
    """
    Generate simulated WiFi CSI-like patient monitoring data.
    This is not real hardware data yet.
    """

    mode = CURRENT_SCENARIO["mode"]

    if mode == "auto":
        mode = random.choice([
            "normal",
            "normal",
            "normal",
            "warning",
            "critical",
            "fall",
            "no_patient",
        ])

    if mode == "no_patient":
        return _build_sample(
            mode=mode,
            presence=False,
            movement_level=0.0,
            breathing_rate=0,
            heart_rate=0,
            signal_strength=0.0,
            scenario="no_patient_detected",
        )

    if mode == "normal":
        return _build_sample(
            mode=mode,
            presence=True,
            movement_level=round(random.uniform(0.35, 0.75), 2),
            breathing_rate=random.randint(12, 20),
            heart_rate=random.randint(65, 95),
            signal_strength=round(random.uniform(0.70, 0.98), 2),
            scenario="normal",
        )

    if mode == "warning":
        return _build_sample(
            mode=mode,
            presence=True,
            movement_level=round(random.uniform(0.10, 0.25), 2),
            breathing_rate=random.choice([
                random.randint(10, 11),
                random.randint(21, 24),
            ]),
            heart_rate=random.randint(95, 115),
            signal_strength=round(random.uniform(0.50, 0.70), 2),
            scenario="breathing_anomaly",
        )

    if mode == "critical":
        return _build_sample(
            mode=mode,
            presence=True,
            movement_level=round(random.uniform(0.02, 0.12), 2),
            breathing_rate=random.choice([
                random.randint(6, 9),
                random.randint(25, 32),
            ]),
            heart_rate=random.choice([
                random.randint(40, 55),
                random.randint(120, 145),
            ]),
            signal_strength=round(random.uniform(0.25, 0.45), 2),
            scenario="critical_vital_pattern",
        )

    if mode == "fall":
        return _build_sample(
            mode=mode,
            presence=True,
            movement_level=round(random.uniform(0.92, 1.0), 2),
            breathing_rate=random.randint(18, 28),
            heart_rate=random.randint(105, 135),
            signal_strength=round(random.uniform(0.20, 0.42), 2),
            scenario="possible_fall",
        )

    return _build_sample(
        mode="normal",
        presence=True,
        movement_level=round(random.uniform(0.35, 0.75), 2),
        breathing_rate=random.randint(12, 20),
        heart_rate=random.randint(65, 95),
        signal_strength=round(random.uniform(0.70, 0.98), 2),
        scenario="normal",
    )
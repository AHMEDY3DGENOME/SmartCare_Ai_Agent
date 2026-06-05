import math
import random
from datetime import datetime


def _noise(level: float = 0.03) -> float:
    return random.uniform(-level, level)


def generate_csi_waveform(mode: str = "normal", points: int = 80) -> dict:
    """
    Generate simulated WiFi CSI waveform data.

    This simulates:
    - CSI amplitude
    - CSI phase
    - subcarrier response
    - breathing micro-motion
    - movement distortion
    - fall spike pattern

    This is simulated data for demo purposes.
    """

    amplitude = []
    phase = []
    subcarriers = []

    base_amplitude = 0.65
    base_phase = 0.0

    breathing_frequency = 0.12
    motion_frequency = 0.35

    for i in range(points):
        t = i / points

        breathing_wave = 0.08 * math.sin(2 * math.pi * breathing_frequency * i)
        motion_wave = 0.04 * math.sin(2 * math.pi * motion_frequency * i)

        value = base_amplitude + breathing_wave + motion_wave + _noise(0.025)
        phase_value = base_phase + 0.3 * math.sin(2 * math.pi * breathing_frequency * i) + _noise(0.03)

        if mode == "normal":
            value += _noise(0.015)

        elif mode == "warning":
            value += 0.08 * math.sin(2 * math.pi * 0.55 * i)
            phase_value += _noise(0.08)

        elif mode == "critical":
            value = 0.35 + 0.18 * math.sin(2 * math.pi * 0.85 * i) + _noise(0.09)
            phase_value = 0.8 * math.sin(2 * math.pi * 0.65 * i) + _noise(0.14)

        elif mode == "fall":
            spike_center = points // 2
            spike_width = 5

            spike = math.exp(-((i - spike_center) ** 2) / (2 * spike_width ** 2))
            value = 0.45 + (0.75 * spike) + _noise(0.08)
            phase_value = 0.25 + (1.2 * spike) + _noise(0.12)

            if i > spike_center + 8:
                value = 0.25 + _noise(0.03)
                phase_value = 0.05 + _noise(0.04)

        elif mode == "no_patient":
            value = 0.08 + _noise(0.015)
            phase_value = _noise(0.02)

        value = max(0.0, min(1.0, round(value, 4)))
        phase_value = round(phase_value, 4)

        amplitude.append(value)
        phase.append(phase_value)
        subcarriers.append({
            "index": i,
            "amplitude": value,
            "phase": phase_value
        })

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "mode": mode,
        "points": points,
        "amplitude": amplitude,
        "phase": phase,
        "subcarriers": subcarriers,
        "description": "Simulated WiFi CSI waveform for non-wearable patient monitoring."
    }
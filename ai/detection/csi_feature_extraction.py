import statistics


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def _safe_variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return round(statistics.variance(values), 4)


def _safe_range(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(max(values) - min(values), 4)


def _count_spikes(values: list[float], threshold: float = 0.25) -> int:
    """
    Count sudden jumps between consecutive CSI points.
    """
    if len(values) < 2:
        return 0

    spikes = 0

    for i in range(1, len(values)):
        if abs(values[i] - values[i - 1]) >= threshold:
            spikes += 1

    return spikes


def extract_csi_features(sensor_data: dict) -> dict:
    """
    Extract interpretable features from simulated WiFi CSI waveform.

    These features help explain what the AI system is detecting:
    - amplitude variance
    - phase instability
    - signal range
    - spike count
    - motion intensity
    - breathing regularity
    """

    wifi_csi = sensor_data.get("wifi_csi", {})

    amplitude = wifi_csi.get("amplitude", [])
    phase = wifi_csi.get("phase", [])

    amplitude_mean = _safe_mean(amplitude)
    amplitude_variance = _safe_variance(amplitude)
    amplitude_range = _safe_range(amplitude)

    phase_mean = _safe_mean(phase)
    phase_variance = _safe_variance(phase)
    phase_range = _safe_range(phase)

    amplitude_spikes = _count_spikes(amplitude, threshold=0.20)
    phase_spikes = _count_spikes(phase, threshold=0.35)

    movement_level = sensor_data.get("movement_level", 0)
    breathing_rate = sensor_data.get("breathing_rate", 0)
    signal_strength = sensor_data.get("signal_strength", 0)
    scenario = sensor_data.get("scenario", "unknown")

    motion_intensity = round(
        (movement_level * 0.5)
        + (amplitude_variance * 2.0)
        + (amplitude_spikes * 0.03),
        4
    )

    phase_instability = round(
        (phase_variance * 2.0)
        + (phase_spikes * 0.04),
        4
    )

    fall_spike_score = round(
        (amplitude_spikes * 8)
        + (phase_spikes * 5)
        + (amplitude_range * 40)
        + ((1 - signal_strength) * 25),
        2
    )

    if 12 <= breathing_rate <= 20 and phase_instability < 0.35:
        breathing_regularity = "regular"
    elif 10 <= breathing_rate <= 24:
        breathing_regularity = "slightly_irregular"
    else:
        breathing_regularity = "irregular"

    if scenario == "possible_fall" or fall_spike_score >= 65:
        csi_pattern = "fall_like_disturbance"
    elif phase_instability >= 0.45:
        csi_pattern = "unstable_phase_pattern"
    elif motion_intensity >= 0.65:
        csi_pattern = "high_motion_pattern"
    elif signal_strength <= 0.35:
        csi_pattern = "weak_signal_pattern"
    else:
        csi_pattern = "stable_monitoring_pattern"

    return {
        "amplitude": {
            "mean": amplitude_mean,
            "variance": amplitude_variance,
            "range": amplitude_range,
            "spikes": amplitude_spikes,
        },
        "phase": {
            "mean": phase_mean,
            "variance": phase_variance,
            "range": phase_range,
            "spikes": phase_spikes,
        },
        "derived_features": {
            "motion_intensity": motion_intensity,
            "phase_instability": phase_instability,
            "fall_spike_score": fall_spike_score,
            "breathing_regularity": breathing_regularity,
            "csi_pattern": csi_pattern,
        },
        "explanation": (
            "Features extracted from simulated WiFi CSI amplitude and phase "
            "to support movement, breathing, and fall-pattern interpretation."
        )
    }
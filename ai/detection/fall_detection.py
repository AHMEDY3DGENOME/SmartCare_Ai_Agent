def detect_fall(sensor_data: dict) -> dict:
    """
    Simulated fall detection based on WiFi CSI patterns.
    """

    movement_level = sensor_data.get("movement_level", 0)
    signal_strength = sensor_data.get("signal_strength", 0)
    scenario = sensor_data.get("scenario", "")

    fall_detected = False
    confidence = 0

    if scenario == "possible_fall":
        fall_detected = True
        confidence = 95

    elif movement_level > 0.90 and signal_strength < 0.45:
        fall_detected = True
        confidence = 85

    return {
        "fall_detected": fall_detected,
        "confidence": confidence
    }
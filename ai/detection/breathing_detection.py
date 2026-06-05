def detect_breathing(sensor_data: dict) -> dict:
    """
    Analyze breathing rate from simulated WiFi sensing data.
    """

    breathing_rate = sensor_data.get("breathing_rate", 0)

    if 12 <= breathing_rate <= 20:
        status = "normal"

    elif 10 <= breathing_rate < 12:
        status = "low"

    elif 21 <= breathing_rate <= 24:
        status = "elevated"

    else:
        status = "abnormal"

    return {
        "breathing_rate": breathing_rate,
        "breathing_status": status
    }
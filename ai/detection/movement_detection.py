def detect_movement(sensor_data: dict) -> dict:
    """
    Analyze movement level from simulated WiFi CSI data.
    """

    movement_level = sensor_data.get("movement_level", 0)

    if movement_level >= 0.70:
        status = "high_activity"

    elif movement_level >= 0.30:
        status = "normal_activity"

    elif movement_level >= 0.10:
        status = "low_activity"

    else:
        status = "no_activity"

    return {
        "movement_level": movement_level,
        "movement_status": status
    }
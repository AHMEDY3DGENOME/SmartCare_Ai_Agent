def calculate_risk_score(sensor_data: dict) -> dict:
    """
    Calculate patient risk level based on simulated WiFi sensing data.
    """

    score = 0
    reasons = []

    presence = sensor_data.get("presence", False)
    movement_level = sensor_data.get("movement_level", 0)
    breathing_rate = sensor_data.get("breathing_rate", 0)
    heart_rate = sensor_data.get("heart_rate", 0)
    signal_strength = sensor_data.get("signal_strength", 0)
    scenario = sensor_data.get("scenario", "unknown")

    if not presence:
        return {
            "risk_score": 0,
            "risk_level": "no_patient",
            "reasons": ["No patient detected in the monitored area."]
        }

    if movement_level < 0.15:
        score += 25
        reasons.append("Low movement detected.")

    if breathing_rate < 12 or breathing_rate > 22:
        score += 30
        reasons.append("Abnormal breathing rate detected.")

    if heart_rate < 60 or heart_rate > 110:
        score += 20
        reasons.append("Heart rate outside normal simulated range.")

    if signal_strength < 0.45:
        score += 15
        reasons.append("Weak or unstable WiFi CSI signal pattern.")

    if scenario == "possible_fall":
        score += 50
        reasons.append("Possible fall pattern detected.")

    if score >= 70:
        risk_level = "critical"
    elif score >= 35:
        risk_level = "warning"
    else:
        risk_level = "normal"

    if not reasons:
        reasons.append("Patient condition appears stable.")

    return {
        "risk_score": score,
        "risk_level": risk_level,
        "reasons": reasons
    }
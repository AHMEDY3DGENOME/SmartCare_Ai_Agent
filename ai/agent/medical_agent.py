def generate_agent_summary(
    sensor_data: dict,
    movement_analysis: dict,
    breathing_analysis: dict,
    fall_analysis: dict,
    risk_assessment: dict
) -> dict:
    risk_level = risk_assessment.get("risk_level", "unknown")
    reasons = risk_assessment.get("reasons", [])

    if risk_level == "critical":
        summary = "Critical risk detected. The patient may need urgent attention."
        action = "Notify the doctor immediately and start emergency check workflow."

    elif risk_level == "warning":
        summary = "Warning signs detected. The patient condition should be monitored closely."
        action = "Continue monitoring and notify clinical staff if the condition persists."

    elif risk_level == "no_patient":
        summary = "No patient is detected in the monitored area."
        action = "Check whether the patient has left the monitored area."

    else:
        summary = "The patient appears stable based on the current simulated WiFi sensing data."
        action = "Continue normal monitoring."

    return {
        "agent_name": "CareSense AI Agent",
        "summary": summary,
        "risk_level": risk_level,
        "reasons": reasons,
        "recommended_action": action,
        "disclaimer": "This is a simulated AI demo and not a medical diagnosis."
    }
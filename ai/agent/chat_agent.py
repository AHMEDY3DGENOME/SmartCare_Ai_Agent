def _normalize_question(question: str) -> str:
    return (question or "").strip().lower()


def _format_reasons(reasons: list) -> str:
    if not reasons:
        return "No specific risk reasons were detected."

    return " ".join([f"- {reason}" for reason in reasons])


def generate_chat_response(question: str, dashboard_payload: dict) -> dict:
    """
    Generate a simple rule-based AI Agent chat response
    based on the latest dashboard payload.

    Later we can replace this with an LLM.
    """

    question_text = _normalize_question(question)

    sensor = dashboard_payload.get("sensor_data", {})
    risk = dashboard_payload.get("risk_assessment", {})
    fall = dashboard_payload.get("fall_analysis", {})
    breathing = dashboard_payload.get("breathing_analysis", {})
    movement = dashboard_payload.get("movement_analysis", {})
    csi_features = dashboard_payload.get("csi_features", {})
    agent = dashboard_payload.get("ai_agent", {})

    derived = csi_features.get("derived_features", {})

    risk_level = risk.get("risk_level", "unknown")
    risk_score = risk.get("risk_score", 0)
    reasons = risk.get("reasons", [])

    presence = sensor.get("presence", False)
    scenario = sensor.get("scenario", "unknown")
    mode = sensor.get("mode", "unknown")

    fall_detected = fall.get("fall_detected", False)
    fall_confidence = fall.get("confidence", 0)

    breathing_rate = sensor.get("breathing_rate", 0)
    breathing_status = breathing.get("breathing_status", "unknown")

    movement_level = sensor.get("movement_level", 0)
    movement_status = movement.get("movement_status", "unknown")

    csi_pattern = derived.get("csi_pattern", "unknown")
    motion_intensity = derived.get("motion_intensity", 0)
    phase_instability = derived.get("phase_instability", 0)
    fall_spike_score = derived.get("fall_spike_score", 0)
    breathing_regularity = derived.get("breathing_regularity", "unknown")

    recommended_action = agent.get(
        "recommended_action",
        "Continue monitoring the patient."
    )

    if not question_text:
        answer = (
            "Please ask a question about the patient's status, fall risk, "
            "breathing pattern, WiFi CSI signal, or recommended action."
        )

    elif any(word in question_text for word in ["status", "condition", "الحالة", "وضع"]):
        if not presence:
            answer = (
                "No patient is currently detected in the monitored area. "
                "Please verify whether the patient has left the room or the sensor zone."
            )
        else:
            answer = (
                f"The patient is currently in {risk_level.upper()} state. "
                f"The risk score is {risk_score}. "
                f"Movement status is {movement_status} with movement level {movement_level}. "
                f"Breathing status is {breathing_status} at {breathing_rate} breaths per minute. "
                f"Recommended action: {recommended_action}"
            )

    elif any(word in question_text for word in ["fall", "سقوط", "وقع", "fallen"]):
        if fall_detected:
            answer = (
                f"Yes, a possible fall pattern was detected. "
                f"The fall confidence is {fall_confidence}%. "
                f"The CSI fall spike score is {fall_spike_score}. "
                f"Recommended action: {recommended_action}"
            )
        else:
            answer = (
                f"No fall is currently detected. "
                f"The fall confidence is {fall_confidence}%. "
                f"The CSI pattern is {csi_pattern}."
            )

    elif any(word in question_text for word in ["breathing", "breath", "تنفس", "نفس"]):
        answer = (
            f"The breathing rate is {breathing_rate} breaths per minute. "
            f"The breathing status is {breathing_status}. "
            f"CSI breathing regularity is {breathing_regularity}. "
            f"Recommended action: {recommended_action}"
        )

    elif any(word in question_text for word in ["movement", "motion", "حركة", "يتحرك"]):
        answer = (
            f"The movement level is {movement_level}. "
            f"The movement status is {movement_status}. "
            f"CSI motion intensity is {motion_intensity}. "
            f"The detected CSI pattern is {csi_pattern}."
        )

    elif any(word in question_text for word in ["wifi", "csi", "signal", "واي", "اشارة", "إشارة"]):
        answer = (
            f"The current WiFi CSI pattern is {csi_pattern}. "
            f"Motion intensity is {motion_intensity}. "
            f"Phase instability is {phase_instability}. "
            f"Fall spike score is {fall_spike_score}. "
            f"This is based on simulated CSI amplitude and phase waveforms."
        )

    elif any(word in question_text for word in ["action", "recommend", "doctor", "اعمل", "الطبيب"]):
        answer = (
            f"Recommended action: {recommended_action} "
            f"Risk level is {risk_level.upper()} with score {risk_score}."
        )

    elif any(word in question_text for word in ["why", "reason", "سبب", "ليه"]):
        answer = (
            f"The current risk level is {risk_level.upper()} because: "
            f"{_format_reasons(reasons)}"
        )

    else:
        answer = (
            f"Current mode is {mode}. "
            f"Scenario is {scenario}. "
            f"Risk level is {risk_level.upper()} with score {risk_score}. "
            f"CSI pattern is {csi_pattern}. "
            f"Recommended action: {recommended_action}"
        )

    return {
        "agent": "CareSense AI Chat Agent",
        "question": question,
        "answer": answer,
        "context": {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "scenario": scenario,
            "mode": mode,
            "fall_detected": fall_detected,
            "csi_pattern": csi_pattern,
        },
        "disclaimer": "This is a simulated AI demo and not a medical diagnosis."
    }
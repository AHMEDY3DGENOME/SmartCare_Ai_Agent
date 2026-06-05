from fastapi import APIRouter

from simulator.generator import (
    generate_wifi_csi_sample,
    set_simulation_mode,
    get_simulation_mode,
)
from ai.risk_engine.risk_score import calculate_risk_score
from ai.detection.movement_detection import detect_movement
from ai.detection.breathing_detection import detect_breathing
from ai.detection.fall_detection import detect_fall
from ai.detection.csi_feature_extraction import extract_csi_features
from ai.agent.medical_agent import generate_agent_summary
from room.digital_twin import build_digital_twin

router = APIRouter()

DASHBOARD_EVENTS = []


def build_dashboard_payload():
    sensor_data = generate_wifi_csi_sample()

    movement = detect_movement(sensor_data)
    breathing = detect_breathing(sensor_data)
    fall = detect_fall(sensor_data)
    risk = calculate_risk_score(sensor_data)
    csi_features = extract_csi_features(sensor_data)

    digital_twin = build_digital_twin(
        sensor_data=sensor_data,
        risk_assessment=risk,
        fall_analysis=fall,
        csi_features=csi_features,
    )

    agent_summary = generate_agent_summary(
        sensor_data=sensor_data,
        movement_analysis=movement,
        breathing_analysis=breathing,
        fall_analysis=fall,
        risk_assessment=risk
    )

    payload = {
        "current_mode": get_simulation_mode(),
        "sensor_data": sensor_data,
        "movement_analysis": movement,
        "breathing_analysis": breathing,
        "fall_analysis": fall,
        "risk_assessment": risk,
        "csi_features": csi_features,
        "digital_twin": digital_twin,
        "ai_agent": agent_summary,
    }

    risk_level = risk.get("risk_level")
    csi_pattern = csi_features.get("derived_features", {}).get("csi_pattern")

    if risk_level in ["warning", "critical"]:
        DASHBOARD_EVENTS.append({
            "timestamp": sensor_data.get("timestamp"),
            "mode": get_simulation_mode(),
            "event_type": risk_level,
            "scenario": sensor_data.get("scenario"),
            "csi_pattern": csi_pattern,
            "patient_activity": digital_twin.get("patient", {}).get("activity"),
            "patient_location": digital_twin.get("patient", {}).get("location"),
            "message": agent_summary.get("summary"),
            "recommended_action": agent_summary.get("recommended_action")
        })

    if len(DASHBOARD_EVENTS) > 20:
        DASHBOARD_EVENTS.pop(0)

    return payload


@router.get("/dashboard/live")
def dashboard_live():
    return build_dashboard_payload()


@router.get("/dashboard/events")
def dashboard_events():
    return {
        "total_events": len(DASHBOARD_EVENTS),
        "events": list(reversed(DASHBOARD_EVENTS))
    }


@router.get("/dashboard/status")
def dashboard_status():
    payload = build_dashboard_payload()

    sensor = payload["sensor_data"]
    risk = payload["risk_assessment"]
    fall = payload["fall_analysis"]
    agent = payload["ai_agent"]
    csi_features = payload["csi_features"]
    digital_twin = payload["digital_twin"]

    return {
        "patient": {
            "name": "Demo Patient",
            "room": "Room 01",
            "monitoring_mode": "Simulated WiFi CSI",
            "current_mode": get_simulation_mode(),
        },
        "status": {
            "presence": sensor.get("presence"),
            "risk_level": risk.get("risk_level"),
            "risk_score": risk.get("risk_score"),
            "fall_detected": fall.get("fall_detected"),
            "agent_summary": agent.get("summary"),
            "recommended_action": agent.get("recommended_action")
        },
        "metrics": {
            "movement_level": sensor.get("movement_level"),
            "breathing_rate": sensor.get("breathing_rate"),
            "heart_rate": sensor.get("heart_rate"),
            "signal_strength": sensor.get("signal_strength"),
            "scenario": sensor.get("scenario"),
            "mode": sensor.get("mode"),
        },
        "csi_features": csi_features,
        "digital_twin": digital_twin,
    }


@router.post("/dashboard/mode/{mode}")
def change_dashboard_mode(mode: str):
    """
    Change the live simulation mode from the dashboard.
    Modes:
    auto, normal, warning, critical, fall, no_patient
    """

    result = set_simulation_mode(mode)

    return result


@router.get("/dashboard/mode")
def dashboard_mode():
    return {
        "current_mode": get_simulation_mode(),
        "available_modes": [
            "auto",
            "normal",
            "warning",
            "critical",
            "fall",
            "no_patient",
        ]
    }


@router.delete("/dashboard/events")
def clear_dashboard_events():
    DASHBOARD_EVENTS.clear()

    return {
        "success": True,
        "message": "Dashboard events cleared."
    }
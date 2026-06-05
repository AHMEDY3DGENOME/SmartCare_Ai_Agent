from fastapi import APIRouter

from simulator.generator import generate_wifi_csi_sample
from ai.risk_engine.risk_score import calculate_risk_score
from ai.detection.movement_detection import detect_movement
from ai.detection.breathing_detection import detect_breathing
from ai.detection.fall_detection import detect_fall
from ai.agent.medical_agent import generate_agent_summary

router = APIRouter()


@router.get("/simulator/live")
def live_simulation():

    sensor_data = generate_wifi_csi_sample()

    movement = detect_movement(sensor_data)
    breathing = detect_breathing(sensor_data)
    fall = detect_fall(sensor_data)
    risk = calculate_risk_score(sensor_data)

    agent_summary = generate_agent_summary(
        sensor_data=sensor_data,
        movement_analysis=movement,
        breathing_analysis=breathing,
        fall_analysis=fall,
        risk_assessment=risk
    )

    return {
        "sensor_data": sensor_data,
        "movement_analysis": movement,
        "breathing_analysis": breathing,
        "fall_analysis": fall,
        "risk_assessment": risk,
        "ai_agent": agent_summary
    }
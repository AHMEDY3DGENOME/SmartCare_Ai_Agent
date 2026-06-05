import random


def determine_patient_activity(
    sensor_data: dict,
    fall_analysis: dict,
    csi_features: dict
) -> str:
    """
    Convert CSI analysis into a human activity state.
    """

    if not sensor_data.get("presence", False):
        return "no_patient"

    if fall_analysis.get("fall_detected", False):
        return "fallen"

    movement_level = sensor_data.get("movement_level", 0)

    if movement_level >= 0.80:
        return "walking"

    if movement_level >= 0.30:
        return "standing"

    if movement_level >= 0.10:
        return "sitting"

    return "sleeping"


def determine_patient_location(activity: str) -> str:
    """
    Simulate location based on detected activity.
    """

    if activity == "sleeping":
        return "bed"

    if activity == "sitting":
        return "chair"

    if activity == "fallen":
        return random.choice([
            "near_bed",
            "near_chair"
        ])

    if activity == "walking":
        return random.choice([
            "center_room",
            "near_bed",
            "near_door",
            "near_chair"
        ])

    if activity == "standing":
        return random.choice([
            "center_room",
            "near_bed"
        ])

    return "unknown"


def determine_avatar_status(
    risk_assessment: dict,
    activity: str
) -> str:
    """
    Avatar visual state.
    """

    if activity == "fallen":
        return "fall_alert"

    risk_level = risk_assessment.get(
        "risk_level",
        "normal"
    )

    if risk_level == "critical":
        return "critical"

    if risk_level == "warning":
        return "warning"

    return "stable"


def build_patient_tracking(
    sensor_data: dict,
    risk_assessment: dict,
    fall_analysis: dict,
    csi_features: dict
) -> dict:

    activity = determine_patient_activity(
        sensor_data,
        fall_analysis,
        csi_features
    )

    location = determine_patient_location(
        activity
    )

    avatar_status = determine_avatar_status(
        risk_assessment,
        activity
    )

    return {
        "activity": activity,
        "location": location,
        "avatar_status": avatar_status,
        "presence": sensor_data.get(
            "presence",
            False
        )
    }
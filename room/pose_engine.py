def get_pose_class(activity: str) -> str:
    """
    Convert detected patient activity into frontend CSS pose class.
    """

    pose_map = {
        "walking": "pose-walking",
        "standing": "pose-standing",
        "sitting": "pose-sitting",
        "sleeping": "pose-sleeping",
        "fallen": "pose-fallen",
        "no_patient": "pose-no-patient",
    }

    return pose_map.get(
        activity,
        "pose-standing"
    )


def get_risk_color_class(
    avatar_status: str,
    risk_level: str
) -> str:
    """
    Convert medical risk state into frontend color class.
    """

    if avatar_status == "fall_alert":
        return "risk-red-pulse"

    if risk_level == "critical":
        return "risk-red"

    if risk_level == "warning":
        return "risk-yellow"

    return "risk-green"


def get_pose_description(activity: str) -> str:
    """
    Human-readable description for the current digital twin pose.
    """

    descriptions = {
        "walking": "Patient appears to be moving inside the room.",
        "standing": "Patient is upright with stable posture.",
        "sitting": "Patient appears to be sitting.",
        "sleeping": "Patient appears to be resting or sleeping.",
        "fallen": "Possible fall posture detected.",
        "no_patient": "No patient detected inside the monitoring area.",
    }

    return descriptions.get(
        activity,
        "Patient posture is being monitored."
    )


def build_pose_state(
    activity: str,
    avatar_status: str,
    risk_level: str
) -> dict:
    """
    Build full pose state used by the frontend digital twin.
    """

    return {
        "pose_class": get_pose_class(activity),
        "risk_color_class": get_risk_color_class(
            avatar_status,
            risk_level
        ),
        "description": get_pose_description(activity)
    }
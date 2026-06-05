from room.room_state import get_room_layout
from room.patient_tracker import build_patient_tracking
from room.pose_engine import build_pose_state


def _get_patient_position(location: str) -> dict:
    """
    Convert logical patient location into x/y coordinates
    inside the digital room.
    """

    positions = {
        "bed": {
            "x": 170,
            "y": 350
        },
        "chair": {
            "x": 620,
            "y": 360
        },
        "near_bed": {
            "x": 330,
            "y": 310
        },
        "near_chair": {
            "x": 510,
            "y": 330
        },
        "near_door": {
            "x": 680,
            "y": 150
        },
        "center_room": {
            "x": 420,
            "y": 240
        },
        "unknown": {
            "x": 400,
            "y": 250
        }
    }

    return positions.get(
        location,
        positions["unknown"]
    )


def _get_avatar_icon(activity: str) -> str:
    """
    Return simple visual avatar based on patient activity.
    Kept for compatibility, even though the frontend now uses skeleton nodes.
    """

    icons = {
        "walking": "🚶",
        "standing": "🧍",
        "sitting": "🪑",
        "sleeping": "😴",
        "fallen": "🚨",
        "no_patient": "—"
    }

    return icons.get(activity, "🧍")


def _get_status_label(avatar_status: str) -> str:
    labels = {
        "stable": "Stable",
        "warning": "Warning",
        "critical": "Critical",
        "fall_alert": "Fall Alert"
    }

    return labels.get(
        avatar_status,
        "Unknown"
    )


def build_digital_twin(
    sensor_data: dict,
    risk_assessment: dict,
    fall_analysis: dict,
    csi_features: dict
) -> dict:
    """
    Build full digital twin state from WiFi CSI analysis.
    """

    room_layout = get_room_layout()

    tracking = build_patient_tracking(
        sensor_data=sensor_data,
        risk_assessment=risk_assessment,
        fall_analysis=fall_analysis,
        csi_features=csi_features
    )

    activity = tracking.get("activity")
    location = tracking.get("location")
    avatar_status = tracking.get("avatar_status")
    risk_level = risk_assessment.get("risk_level", "normal")

    position = _get_patient_position(
        location
    )

    pose_state = build_pose_state(
        activity=activity,
        avatar_status=avatar_status,
        risk_level=risk_level
    )

    vitals = {
        "temperature": sensor_data.get(
            "temperature",
            36.8
        ),
        "heart_rate": sensor_data.get(
            "heart_rate",
            0
        ),
        "breathing_rate": sensor_data.get(
            "breathing_rate",
            0
        ),
        "oxygen_saturation": sensor_data.get(
            "oxygen_saturation",
            98
        )
    }

    return {
        "room": room_layout,
        "patient": {
            "id": "demo_patient_01",
            "name": "Demo Patient",
            "activity": activity,
            "location": location,
            "position": position,
            "pose_state": pose_state,
            "avatar": {
                "icon": _get_avatar_icon(activity),
                "status": avatar_status,
                "status_label": _get_status_label(
                    avatar_status
                )
            },
            "presence": tracking.get(
                "presence",
                False
            ),
            "vitals": vitals
        },
        "source": {
            "type": "simulated_wifi_csi",
            "description": (
                "Patient movement, posture, location and vital state "
                "are inferred from simulated WiFi CSI waveform analysis."
            )
        }
    }
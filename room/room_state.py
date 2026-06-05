ROOM_LAYOUT = {
    "room_id": "demo_room_01",
    "name": "CareSense Demo Patient Room",
    "width": 800,
    "height": 500,
    "zones": {
        "bed": {
            "label": "Bed",
            "x": 80,
            "y": 300,
            "width": 220,
            "height": 120
        },
        "chair": {
            "label": "Chair",
            "x": 570,
            "y": 320,
            "width": 100,
            "height": 90
        },
        "door": {
            "label": "Door",
            "x": 700,
            "y": 60,
            "width": 70,
            "height": 120
        },
        "monitoring_zone": {
            "label": "WiFi CSI Monitoring Zone",
            "x": 40,
            "y": 40,
            "width": 720,
            "height": 400
        }
    },
    "anchors": {
        "wifi_router": {
            "label": "WiFi CSI Router",
            "x": 40,
            "y": 40
        },
        "receiver": {
            "label": "CSI Receiver",
            "x": 760,
            "y": 440
        }
    }
}


def get_room_layout() -> dict:
    """
    Return static room layout used by the digital twin dashboard.
    """

    return ROOM_LAYOUT
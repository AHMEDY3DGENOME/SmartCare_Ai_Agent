from datetime import datetime
from typing import Dict, Any, List, Optional


class CareSenseClinicalAgent:
    """
    CareSense Clinical AI Agent

    This agent is designed to behave like an intelligent medical monitoring
    assistant on top of simulated WiFi CSI sensing data.

    It does not only answer by keywords. It:
    - understands the user's intent
    - reads the latest monitoring payload
    - selects an internal tool
    - performs clinical-style reasoning
    - returns structured output for UI / voice / future Odoo actions
    """

    def __init__(self):
        self.agent_name = "CareSense Clinical AI Agent"

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, question: str, dashboard_payload: Dict[str, Any]) -> Dict[str, Any]:
        question_text = self._normalize(question)

        snapshot = self._extract_snapshot(dashboard_payload)
        intent = self._detect_intent(question_text)
        tool_used = self._select_tool(intent)

        if not question_text:
            result = self._handle_empty_question(snapshot)

        elif intent == "patient_status":
            result = self._tool_patient_status(snapshot)

        elif intent == "vital_signs":
            result = self._tool_vital_signs(snapshot)

        elif intent == "fall_detection":
            result = self._tool_fall_detection(snapshot)

        elif intent == "breathing_analysis":
            result = self._tool_breathing_analysis(snapshot)

        elif intent == "movement_analysis":
            result = self._tool_movement_analysis(snapshot)

        elif intent == "wifi_csi_analysis":
            result = self._tool_wifi_csi_analysis(snapshot)

        elif intent == "risk_reasoning":
            result = self._tool_risk_reasoning(snapshot)

        elif intent == "recommended_action":
            result = self._tool_recommended_action(snapshot)

        elif intent == "location":
            result = self._tool_location(snapshot)

        elif intent == "emergency":
            result = self._tool_emergency(snapshot)

        else:
            result = self._tool_general_clinical_answer(snapshot, question_text)

        return {
            "agent": self.agent_name,
            "question": question,
            "intent": intent,
            "tool_used": tool_used,
            "severity": result.get("severity", snapshot["risk_level"]),
            "answer": result.get("answer"),
            "recommended_action": result.get("recommended_action"),
            "clinical_summary": result.get("clinical_summary"),
            "context": {
                "risk_level": snapshot["risk_level"],
                "risk_score": snapshot["risk_score"],
                "scenario": snapshot["scenario"],
                "mode": snapshot["mode"],
                "presence": snapshot["presence"],
                "fall_detected": snapshot["fall_detected"],
                "fall_confidence": snapshot["fall_confidence"],
                "csi_pattern": snapshot["csi_pattern"],
                "breathing_rate": snapshot["breathing_rate"],
                "heart_rate": snapshot["heart_rate"],
                "movement_level": snapshot["movement_level"],
                "location": snapshot["location"],
                "activity": snapshot["activity"],
            },
            "agent_trace": {
                "timestamp": datetime.utcnow().isoformat(),
                "intent_detected": intent,
                "tool_selected": tool_used,
                "data_source": "live_dashboard_payload",
                "next_odoo_action": result.get("next_odoo_action"),
            },
            "disclaimer": (
                "This is a simulated AI monitoring demo. "
                "It is not a medical diagnosis and must be verified by clinical staff."
            ),
        }

    # ------------------------------------------------------------------
    # Snapshot extraction
    # ------------------------------------------------------------------

    def _extract_snapshot(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        sensor = payload.get("sensor_data", {}) or {}
        risk = payload.get("risk_assessment", {}) or {}
        fall = payload.get("fall_analysis", {}) or {}
        breathing = payload.get("breathing_analysis", {}) or {}
        movement = payload.get("movement_analysis", {}) or {}
        csi_features = payload.get("csi_features", {}) or {}
        agent = payload.get("ai_agent", {}) or {}
        twin = payload.get("digital_twin", {}) or {}

        derived = csi_features.get("derived_features", {}) or {}
        patient = twin.get("patient", {}) or {}
        vitals = patient.get("vitals", {}) or {}

        return {
            "mode": sensor.get("mode", "unknown"),
            "scenario": sensor.get("scenario", "unknown"),
            "timestamp": sensor.get("timestamp"),

            "presence": sensor.get("presence", False),

            "heart_rate": self._safe_number(
                sensor.get("heart_rate", vitals.get("heart_rate", 0))
            ),
            "breathing_rate": self._safe_number(
                sensor.get("breathing_rate", vitals.get("breathing_rate", 0))
            ),
            "temperature": self._safe_number(vitals.get("temperature", 0)),
            "oxygen_saturation": self._safe_number(
                vitals.get("oxygen_saturation", vitals.get("oxygen", 0))
            ),

            "movement_level": self._safe_number(sensor.get("movement_level", 0)),
            "movement_status": movement.get("movement_status", "unknown"),

            "breathing_status": breathing.get("breathing_status", "unknown"),
            "breathing_regularity": derived.get("breathing_regularity", "unknown"),

            "fall_detected": fall.get("fall_detected", False),
            "fall_confidence": self._safe_number(fall.get("confidence", 0)),
            "fall_spike_score": self._safe_number(derived.get("fall_spike_score", 0)),

            "risk_level": risk.get("risk_level", "unknown"),
            "risk_score": self._safe_number(risk.get("risk_score", 0)),
            "risk_reasons": risk.get("reasons", []) or [],

            "csi_pattern": derived.get("csi_pattern", "unknown"),
            "motion_intensity": self._safe_number(derived.get("motion_intensity", 0)),
            "phase_instability": self._safe_number(derived.get("phase_instability", 0)),

            "activity": patient.get("activity", "unknown"),
            "location": patient.get("location", "unknown"),
            "patient_status_label": (
                patient.get("avatar", {}) or {}
            ).get("status_label", "unknown"),

            "summary": agent.get("summary", ""),
            "recommended_action": agent.get(
                "recommended_action",
                "Continue monitoring the patient."
            ),
        }

    # ------------------------------------------------------------------
    # Intent detection
    # ------------------------------------------------------------------

    def _detect_intent(self, question: str) -> str:
        if not question:
            return "empty"

        intent_keywords = {
            "emergency": [
                "emergency", "urgent", "critical", "call doctor",
                "call nurse", "اسعاف", "طوارئ", "خطر", "حرج"
            ],
            "fall_detection": [
                "fall", "fallen", "fell", "سقوط", "وقع", "وقعت", "طاح"
            ],
            "breathing_analysis": [
                "breathing", "breath", "respiration", "تنفس", "النفس", "بيتنفس"
            ],
            "vital_signs": [
                "vitals", "vital", "heart", "pulse", "oxygen", "spo2",
                "temperature", "علامات", "حيوية", "نبض", "حرارة", "اكسجين", "أكسجين"
            ],
            "movement_analysis": [
                "movement", "motion", "activity", "moving",
                "حركة", "يتحرك", "النشاط"
            ],
            "wifi_csi_analysis": [
                "wifi", "wi-fi", "csi", "signal", "amplitude", "phase",
                "واي", "وايفاي", "اشارة", "إشارة", "موجات"
            ],
            "risk_reasoning": [
                "why", "reason", "because", "explain", "risk reason",
                "ليه", "لماذا", "سبب", "تفسير"
            ],
            "recommended_action": [
                "action", "recommend", "what should", "what do",
                "اعمل", "نعمل", "التوصية", "الإجراء", "الطبيب"
            ],
            "location": [
                "where", "location", "room", "bed", "chair",
                "فين", "مكان", "الغرفة", "السرير"
            ],
            "patient_status": [
                "status", "condition", "state", "how is",
                "الحالة", "وضع", "عامل ايه", "عامل إيه"
            ],
        }

        for intent, keywords in intent_keywords.items():
            if any(keyword in question for keyword in keywords):
                return intent

        return "general"

    def _select_tool(self, intent: str) -> str:
        tool_map = {
            "empty": "help_tool",
            "patient_status": "patient_status_tool",
            "vital_signs": "vital_signs_tool",
            "fall_detection": "fall_detection_tool",
            "breathing_analysis": "breathing_analysis_tool",
            "movement_analysis": "movement_analysis_tool",
            "wifi_csi_analysis": "wifi_csi_signal_tool",
            "risk_reasoning": "risk_reasoning_tool",
            "recommended_action": "care_recommendation_tool",
            "location": "patient_location_tool",
            "emergency": "emergency_triage_tool",
            "general": "general_clinical_reasoning_tool",
        }

        return tool_map.get(intent, "general_clinical_reasoning_tool")

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def _handle_empty_question(self, s: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "severity": "info",
            "answer": (
                "You can ask me about the patient's status, vital signs, fall risk, "
                "breathing pattern, movement, WiFi CSI signal, location, or recommended action."
            ),
            "recommended_action": "Ask a monitoring question.",
            "clinical_summary": "Waiting for user question.",
            "next_odoo_action": None,
        }

    def _tool_patient_status(self, s: Dict[str, Any]) -> Dict[str, Any]:
        if not s["presence"]:
            answer = (
                "No patient is currently detected in the monitored area. "
                "The system should verify whether the patient left the room, moved outside "
                "the WiFi sensing zone, or whether there is a sensor coverage issue."
            )

            return self._result(
                s,
                answer,
                "Check patient presence and room coverage.",
                "no_patient",
                "create_presence_check_alert"
            )

        clinical_summary = self._build_clinical_summary(s)

        answer = (
            f"The patient is currently classified as {s['risk_level'].upper()}. "
            f"Current activity is {s['activity']} and location is {s['location']}. "
            f"Heart rate is {s['heart_rate']} bpm, breathing rate is "
            f"{s['breathing_rate']} breaths per minute, oxygen saturation is "
            f"{s['oxygen_saturation']}%, and temperature is {s['temperature']} °C. "
            f"{self._risk_sentence(s)} "
            f"Recommended action: {self._recommended_action_from_snapshot(s)}"
        )

        return self._result(
            s,
            answer,
            self._recommended_action_from_snapshot(s),
            s["risk_level"],
            self._odoo_action_from_risk(s),
            clinical_summary
        )

    def _tool_vital_signs(self, s: Dict[str, Any]) -> Dict[str, Any]:
        flags = self._vital_flags(s)

        if flags:
            flag_text = " ".join(flags)
        else:
            flag_text = "No major simulated vital sign abnormality is currently detected."

        answer = (
            "Current vital signs are: "
            f"heart rate {s['heart_rate']} bpm, "
            f"breathing rate {s['breathing_rate']} breaths per minute, "
            f"oxygen saturation {s['oxygen_saturation']}%, "
            f"and temperature {s['temperature']} °C. "
            f"{flag_text} "
            f"Overall risk level is {s['risk_level'].upper()}."
        )

        return self._result(
            s,
            answer,
            self._recommended_action_from_snapshot(s),
            s["risk_level"],
            self._odoo_action_from_risk(s),
        )

    def _tool_fall_detection(self, s: Dict[str, Any]) -> Dict[str, Any]:
        if s["fall_detected"] or s["fall_spike_score"] >= 50:
            answer = (
                "A possible fall event is detected from the simulated WiFi CSI pattern. "
                f"Fall confidence is {s['fall_confidence']}%, and the CSI fall spike score "
                f"is {s['fall_spike_score']}. The patient activity is currently "
                f"{s['activity']} near {s['location']}. "
                "This should trigger immediate human verification."
            )

            return self._result(
                s,
                answer,
                "Notify clinical staff immediately and verify the patient physically.",
                "critical",
                "create_fall_alert"
            )

        answer = (
            "No active fall event is currently detected. "
            f"Fall confidence is {s['fall_confidence']}%, fall spike score is "
            f"{s['fall_spike_score']}, and the current CSI pattern is {s['csi_pattern']}."
        )

        return self._result(
            s,
            answer,
            "Continue monitoring for sudden movement or fall-like CSI spikes.",
            s["risk_level"],
            None
        )

    def _tool_breathing_analysis(self, s: Dict[str, Any]) -> Dict[str, Any]:
        breathing_rate = s["breathing_rate"]
        status = s["breathing_status"]
        regularity = s["breathing_regularity"]

        if breathing_rate == 0 or not s["presence"]:
            interpretation = (
                "Breathing cannot be reliably interpreted because no patient is detected "
                "or the breathing signal is unavailable."
            )
        elif breathing_rate < 10:
            interpretation = (
                "The simulated breathing rate appears low and should be monitored closely."
            )
        elif breathing_rate > 24:
            interpretation = (
                "The simulated breathing rate appears elevated and may require clinical review."
            )
        else:
            interpretation = (
                "The simulated breathing rate is within a generally acceptable monitoring range."
            )

        answer = (
            f"Breathing rate is {breathing_rate} breaths per minute. "
            f"Breathing status is {status}. "
            f"CSI breathing regularity is {regularity}. "
            f"{interpretation}"
        )

        return self._result(
            s,
            answer,
            self._recommended_action_from_snapshot(s),
            s["risk_level"],
            self._odoo_action_from_risk(s),
        )

    def _tool_movement_analysis(self, s: Dict[str, Any]) -> Dict[str, Any]:
        answer = (
            f"Movement level is {s['movement_level']}. "
            f"Movement status is {s['movement_status']}. "
            f"CSI motion intensity is {s['motion_intensity']}. "
            f"The patient activity is interpreted as {s['activity']}."
        )

        if s["activity"] == "fallen":
            answer += " This activity is consistent with a possible fall posture."

        return self._result(
            s,
            answer,
            self._recommended_action_from_snapshot(s),
            s["risk_level"],
            self._odoo_action_from_risk(s),
        )

    def _tool_wifi_csi_analysis(self, s: Dict[str, Any]) -> Dict[str, Any]:
        answer = (
            f"The current WiFi CSI pattern is {s['csi_pattern']}. "
            f"Motion intensity is {s['motion_intensity']}, phase instability is "
            f"{s['phase_instability']}, and fall spike score is {s['fall_spike_score']}. "
            "The agent is using these simulated CSI-derived features to infer presence, "
            "movement, breathing pattern, and fall-like disturbances."
        )

        return self._result(
            s,
            answer,
            "Use CSI trends together with risk engine output before triggering care workflow.",
            s["risk_level"],
            None
        )

    def _tool_risk_reasoning(self, s: Dict[str, Any]) -> Dict[str, Any]:
        reasons = s["risk_reasons"]

        if reasons:
            reason_text = " ".join([f"- {reason}" for reason in reasons])
        else:
            reason_text = (
                "No specific risk reasons were returned by the risk engine. "
                "The current classification is mainly based on the simulated signal state."
            )

        answer = (
            f"The current risk level is {s['risk_level'].upper()} with score "
            f"{s['risk_score']}. Main reasons: {reason_text}"
        )

        return self._result(
            s,
            answer,
            self._recommended_action_from_snapshot(s),
            s["risk_level"],
            self._odoo_action_from_risk(s),
        )

    def _tool_recommended_action(self, s: Dict[str, Any]) -> Dict[str, Any]:
        action = self._recommended_action_from_snapshot(s)

        answer = (
            f"Recommended action: {action} "
            f"The current risk level is {s['risk_level'].upper()} with score "
            f"{s['risk_score']}."
        )

        return self._result(
            s,
            answer,
            action,
            s["risk_level"],
            self._odoo_action_from_risk(s),
        )

    def _tool_location(self, s: Dict[str, Any]) -> Dict[str, Any]:
        answer = (
            f"The patient is currently interpreted as being at location: "
            f"{s['location']}. Current activity is {s['activity']}."
        )

        return self._result(
            s,
            answer,
            "Use the digital twin view to verify patient position.",
            s["risk_level"],
            None
        )

    def _tool_emergency(self, s: Dict[str, Any]) -> Dict[str, Any]:
        if s["risk_level"] == "critical" or s["fall_detected"]:
            answer = (
                "Emergency-level monitoring response is recommended. "
                "The system detected critical risk or a possible fall pattern. "
                "A nurse or doctor should verify the patient immediately."
            )

            return self._result(
                s,
                answer,
                "Trigger emergency workflow and create a critical alert.",
                "critical",
                "create_emergency_alert"
            )

        answer = (
            "The current data does not indicate an emergency-level event. "
            f"Risk level is {s['risk_level'].upper()}. "
            "Continue monitoring and escalate if the condition changes."
        )

        return self._result(
            s,
            answer,
            self._recommended_action_from_snapshot(s),
            s["risk_level"],
            self._odoo_action_from_risk(s),
        )

    def _tool_general_clinical_answer(
        self,
        s: Dict[str, Any],
        question_text: str
    ) -> Dict[str, Any]:
        answer = (
            "I reviewed the latest patient monitoring snapshot. "
            f"The patient risk level is {s['risk_level'].upper()}, "
            f"risk score is {s['risk_score']}, "
            f"activity is {s['activity']}, "
            f"location is {s['location']}, "
            f"CSI pattern is {s['csi_pattern']}. "
            f"Recommended action: {self._recommended_action_from_snapshot(s)}"
        )

        return self._result(
            s,
            answer,
            self._recommended_action_from_snapshot(s),
            s["risk_level"],
            self._odoo_action_from_risk(s),
        )

    # ------------------------------------------------------------------
    # Reasoning helpers
    # ------------------------------------------------------------------

    def _result(
        self,
        s: Dict[str, Any],
        answer: str,
        recommended_action: str,
        severity: str,
        next_odoo_action: Optional[str],
        clinical_summary: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "answer": answer,
            "recommended_action": recommended_action,
            "severity": severity,
            "clinical_summary": clinical_summary or self._build_clinical_summary(s),
            "next_odoo_action": next_odoo_action,
        }

    def _build_clinical_summary(self, s: Dict[str, Any]) -> str:
        return (
            f"Presence: {s['presence']}. "
            f"Risk: {s['risk_level']} ({s['risk_score']}). "
            f"Activity: {s['activity']}. "
            f"Location: {s['location']}. "
            f"HR: {s['heart_rate']} bpm. "
            f"BR: {s['breathing_rate']} /min. "
            f"SpO2: {s['oxygen_saturation']}%. "
            f"Temp: {s['temperature']} °C. "
            f"CSI: {s['csi_pattern']}."
        )

    def _risk_sentence(self, s: Dict[str, Any]) -> str:
        if s["risk_level"] == "critical":
            return "The system indicates a critical risk state that needs immediate verification."
        if s["risk_level"] == "warning":
            return "The system indicates warning signs that need closer observation."
        if s["risk_level"] == "no_patient":
            return "The system does not currently detect a patient in the sensing area."
        if s["risk_level"] == "normal":
            return "The patient appears stable in the current simulated monitoring snapshot."
        return "The risk state is currently unknown."

    def _recommended_action_from_snapshot(self, s: Dict[str, Any]) -> str:
        if not s["presence"]:
            return "Check whether the patient left the monitored area or if sensor coverage was lost."

        if s["fall_detected"] or s["fall_spike_score"] >= 50:
            return "Notify clinical staff immediately and verify possible fall."

        if s["risk_level"] == "critical":
            return "Notify the doctor immediately and start emergency check workflow."

        if s["risk_level"] == "warning":
            return "Continue close monitoring and notify staff if the warning state persists."

        return s["recommended_action"] or "Continue normal monitoring."

    def _odoo_action_from_risk(self, s: Dict[str, Any]) -> Optional[str]:
        if not s["presence"]:
            return "create_presence_check_alert"

        if s["fall_detected"] or s["fall_spike_score"] >= 50:
            return "create_fall_alert"

        if s["risk_level"] == "critical":
            return "create_critical_patient_alert"

        if s["risk_level"] == "warning":
            return "create_warning_patient_alert"

        return None

    def _vital_flags(self, s: Dict[str, Any]) -> List[str]:
        flags = []

        hr = s["heart_rate"]
        br = s["breathing_rate"]
        spo2 = s["oxygen_saturation"]
        temp = s["temperature"]

        if hr and hr < 50:
            flags.append("Heart rate appears low in this simulated snapshot.")

        if hr and hr > 120:
            flags.append("Heart rate appears elevated in this simulated snapshot.")

        if br and br < 10:
            flags.append("Breathing rate appears low in this simulated snapshot.")

        if br and br > 24:
            flags.append("Breathing rate appears elevated in this simulated snapshot.")

        if spo2 and spo2 < 92:
            flags.append("Oxygen saturation appears low in this simulated snapshot.")

        if temp and temp > 38:
            flags.append("Temperature appears elevated in this simulated snapshot.")

        return flags

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _normalize(self, text: str) -> str:
        return (text or "").strip().lower()

    def _safe_number(self, value: Any) -> float:
        try:
            if value is None:
                return 0
            return round(float(value), 4)
        except (TypeError, ValueError):
            return 0


def generate_chat_response(question: str, dashboard_payload: dict) -> dict:
    """
    Backward-compatible function used by the existing /chat endpoint.
    """
    agent = CareSenseClinicalAgent()
    return agent.run(question, dashboard_payload)
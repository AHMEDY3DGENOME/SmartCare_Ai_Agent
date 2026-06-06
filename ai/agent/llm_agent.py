import os
import json
from datetime import datetime
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from google import genai


load_dotenv()


class CareSenseLLMAgent:
    """
    Real conversational LLM-based CareSense AI Agent using Gemini.

    The agent can:
    - answer normal conversational questions
    - speak Arabic or English based on the user's language
    - analyze live WiFi CSI monitoring data when relevant
    - prepare for future Odoo tool integration
    """

    def __init__(self):
        self.agent_name = "CareSense Gemini Medical AI Agent"
        self.model = os.getenv("LLM_MODEL", "gemini-2.5-flash")
        self.api_key = os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is missing from .env file")

        self.client = genai.Client(api_key=self.api_key)

    def run(self, question: str, dashboard_payload: Dict[str, Any]) -> Dict[str, Any]:
        question = (question or "").strip()
        intent = self._detect_intent(question)
        context = self._extract_context(dashboard_payload)

        prompt = self._build_prompt(
            question=question,
            intent=intent,
            context=context,
            dashboard_payload=dashboard_payload,
        )

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
            )

            answer = response.text or "No response returned from Gemini."

            return {
                "agent": self.agent_name,
                "provider": "gemini",
                "model": self.model,
                "question": question,
                "intent": intent,
                "answer": answer,
                "context": context,
                "agent_trace": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "reasoning_mode": "llm_router",
                    "data_source": self._data_source_for_intent(intent),
                    "odoo_enabled": False,
                    "next_stage": self._next_stage_for_intent(intent),
                },
                "disclaimer": (
                    "This is a simulated AI monitoring demo. "
                    "It is not a medical diagnosis and must be verified by clinical staff."
                ),
            }

        except Exception as e:
            return {
                "agent": self.agent_name,
                "provider": "gemini",
                "model": self.model,
                "question": question,
                "intent": intent,
                "answer": (
                    "CareSense AI could not reach the Gemini model. "
                    f"Technical error: {str(e)}"
                ),
                "error": str(e),
                "context": context,
                "disclaimer": (
                    "This is a simulated AI monitoring demo. "
                    "It is not a medical diagnosis."
                ),
            }

    # ------------------------------------------------------------
    # Intent router
    # ------------------------------------------------------------

    def _detect_intent(self, question: str) -> str:
        q = (question or "").lower().strip()

        if not q:
            return "empty"

        general_keywords = [
            "hello", "hi", "hey", "good morning", "good evening",
            "who are you", "what are you", "can you speak", "speak arabic",
            "arabic", "english", "help", "مرحبا", "اهلا", "أهلا",
            "السلام", "تتكلم عربي", "عربي", "انت مين", "ما اسمك"
        ]

        odoo_keywords = [
            "odoo", "appointment", "book", "patient record", "medical record",
            "consultation", "invoice", "حجز", "موعد", "اودو", "أودو",
            "ملف المريض", "السجل الطبي", "استشارة"
        ]

        emergency_keywords = [
            "emergency", "urgent", "critical", "call doctor", "call nurse",
            "should i call", "doctor now", "اسعاف", "إسعاف", "طوارئ",
            "خطر", "حرج", "اكلم الدكتور", "اتصل بالدكتور"
        ]

        patient_keywords = [
            "patient", "status", "condition", "vitals", "vital signs",
            "heart", "pulse", "breathing", "oxygen", "temperature",
            "risk", "fall", "fallen", "movement", "motion", "location",
            "monitoring summary", "clinical summary",
            "المريض", "الحالة", "علامات", "حيوية", "نبض", "تنفس",
            "أكسجين", "اكسجين", "حرارة", "خطر", "سقوط", "وقع",
            "حركة", "مكان"
        ]

        wifi_keywords = [
            "wifi", "wi-fi", "csi", "signal", "amplitude", "phase",
            "wireless", "موجات", "واي فاي", "وايفاي", "إشارة", "اشارة"
        ]

        if any(k in q for k in emergency_keywords):
            return "emergency"

        if any(k in q for k in odoo_keywords):
            return "odoo"

        if any(k in q for k in wifi_keywords):
            return "wifi_csi"

        if any(k in q for k in patient_keywords):
            return "patient_monitoring"

        if any(k in q for k in general_keywords):
            return "general_chat"

        return "general_chat"

    # ------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------

    def _build_prompt(
        self,
        question: str,
        intent: str,
        context: Dict[str, Any],
        dashboard_payload: Dict[str, Any],
    ) -> str:
        language_rule = """
Language rule:
- If the user writes in Arabic, reply in Arabic.
- If the user writes in English, reply in English.
- Keep the tone professional, clear, and conversational.
"""

        identity = """
You are CareSense AI, a conversational medical monitoring AI agent.

You are part of a non-wearable patient monitoring platform that uses simulated WiFi CSI sensing.
You can discuss:
- patient monitoring
- WiFi CSI sensing
- fall detection
- vital signs
- risk analysis
- future Odoo healthcare integration
- voice assistant workflow

Important safety rules:
- Do not claim to provide a real medical diagnosis.
- Do not invent patient data.
- Only use live monitoring data when the user's question is about the patient, monitoring, risk, fall, vitals, WiFi CSI, or emergency.
- For normal conversation, do not force a clinical report.
"""

        if intent == "empty":
            return f"""
{identity}
{language_rule}

The user sent an empty question.

Reply briefly and ask what they want to know.
"""

        if intent == "general_chat":
            return f"""
{identity}
{language_rule}

User question:
{question}

The user's question is general conversation, not a request for patient analysis.

Answer naturally as CareSense AI.
Do not analyze patient data.
Do not output Clinical Assessment sections unless the user asks about the patient.
"""

        if intent == "odoo":
            return f"""
{identity}
{language_rule}

User question:
{question}

The user is asking about Odoo or healthcare workflow integration.

Current Odoo status:
- Odoo tools are planned but not connected yet.
- Future tools will include:
  1. search_patient_in_odoo
  2. get_patient_medical_record
  3. create_patient_alert
  4. book_appointment
  5. create_clinical_note

Answer as an AI agent that is ready for Odoo integration.
Explain what you will be able to do once Odoo is connected.
Do not pretend that you already fetched real Odoo data.
"""

        if intent == "wifi_csi":
            monitoring_context = self._safe_json(context)

            return f"""
{identity}
{language_rule}

User question:
{question}

The user is asking about WiFi CSI or signal analysis.

Use this live monitoring context:
{monitoring_context}

Answer with:
1. What the WiFi CSI signal indicates
2. How it relates to movement / breathing / fall risk
3. Any limitation of the current simulated data

Do not invent values outside the context.
"""

        if intent == "emergency":
            monitoring_context = self._safe_json(context)

            return f"""
{identity}
{language_rule}

User question:
{question}

The user is asking about emergency action.

Use this live monitoring context:
{monitoring_context}

Give a clear triage-style response:
1. Current risk state
2. Evidence from the data
3. Whether immediate human verification is needed
4. Recommended next action

If risk_level is critical, fall_detected is true, breathing is abnormal, or heart rate is very high,
recommend immediate human verification.
"""

        if intent == "patient_monitoring":
            monitoring_context = self._safe_json(context)

            return f"""
{identity}
{language_rule}

User question:
{question}

The user is asking about the patient or monitoring status.

Use ONLY this live monitoring context:
{monitoring_context}

Answer in a concise clinical dashboard style.

Use this structure only for patient monitoring questions:

Clinical Assessment:
...

Current Evidence:
...

Risk Interpretation:
...

Recommended Action:
...

Important Note:
This is simulated monitoring data and not a medical diagnosis.

Do not mention fields that are missing or None unless relevant.
"""

        return f"""
{identity}
{language_rule}

User question:
{question}

Answer naturally and helpfully.
"""

    # ------------------------------------------------------------
    # Context extraction
    # ------------------------------------------------------------

    def _extract_context(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        sensor = payload.get("sensor_data", {}) or {}
        risk = payload.get("risk_assessment", {}) or {}
        fall = payload.get("fall_analysis", {}) or {}
        breathing = payload.get("breathing_analysis", {}) or {}
        movement = payload.get("movement_analysis", {}) or {}
        csi = payload.get("csi_features", {}) or {}
        twin = payload.get("digital_twin", {}) or {}
        ai_agent = payload.get("ai_agent", {}) or {}

        derived = csi.get("derived_features", {}) or {}
        patient = twin.get("patient", {}) or {}
        vitals = patient.get("vitals", {}) or {}

        return {
            "mode": sensor.get("mode"),
            "scenario": sensor.get("scenario"),
            "timestamp": sensor.get("timestamp"),

            "presence": sensor.get("presence"),
            "movement_level": sensor.get("movement_level"),
            "movement_status": movement.get("movement_status"),

            "breathing_rate": sensor.get("breathing_rate"),
            "breathing_status": breathing.get("breathing_status"),
            "breathing_regularity": derived.get("breathing_regularity"),

            "heart_rate": sensor.get("heart_rate"),
            "oxygen_saturation": vitals.get("oxygen_saturation"),
            "temperature": vitals.get("temperature"),

            "fall_detected": fall.get("fall_detected"),
            "fall_confidence": fall.get("confidence"),
            "fall_spike_score": derived.get("fall_spike_score"),

            "risk_level": risk.get("risk_level"),
            "risk_score": risk.get("risk_score"),
            "risk_reasons": risk.get("reasons"),

            "csi_pattern": derived.get("csi_pattern"),
            "motion_intensity": derived.get("motion_intensity"),
            "phase_instability": derived.get("phase_instability"),

            "activity": patient.get("activity"),
            "location": patient.get("location"),
            "patient_status_label": (
                patient.get("avatar", {}) or {}
            ).get("status_label"),

            "system_recommended_action": ai_agent.get("recommended_action"),
            "system_summary": ai_agent.get("summary"),
        }

    # ------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------

    def _data_source_for_intent(self, intent: str) -> str:
        if intent in ["patient_monitoring", "wifi_csi", "emergency"]:
            return "live_dashboard_payload"
        if intent == "odoo":
            return "planned_odoo_tools"
        return "conversation_only"

    def _next_stage_for_intent(self, intent: str) -> Optional[str]:
        if intent == "odoo":
            return "connect_odoo_tools"
        if intent in ["patient_monitoring", "wifi_csi", "emergency"]:
            return "add_tool_calling_layer"
        return None

    def _safe_json(self, data: Dict[str, Any]) -> str:
        try:
            return json.dumps(data, indent=2, ensure_ascii=False, default=str)
        except Exception:
            return str(data)


def generate_llm_chat_response(question: str, dashboard_payload: dict) -> dict:
    agent = CareSenseLLMAgent()
    return agent.run(question, dashboard_payload)
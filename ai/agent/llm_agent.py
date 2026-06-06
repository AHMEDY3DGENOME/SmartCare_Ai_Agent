import os
import json
import requests
from datetime import datetime
from typing import Dict, Any, Optional

from dotenv import load_dotenv

try:
    from google import genai
except Exception:
    genai = None


load_dotenv()


class CareSenseLLMAgent:
    def __init__(self):
        self.agent_name = "CareSense Medical AI Agent"

        self.provider = os.getenv("LLM_PROVIDER", "auto").lower()

        self.gemini_model = os.getenv("LLM_MODEL", "gemini-2.5-flash")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")

        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        self.ollama_base_url = os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434"
        )

        self.gemini_client = None

        if self.gemini_api_key and genai:
            self.gemini_client = genai.Client(api_key=self.gemini_api_key)

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

        provider_used = None
        model_used = None
        error_message = None

        try:
            if self.provider == "gemini":
                answer = self._call_gemini(prompt)
                provider_used = "gemini"
                model_used = self.gemini_model

            elif self.provider == "ollama":
                answer = self._call_ollama(prompt)
                provider_used = "ollama"
                model_used = self.ollama_model

            else:
                try:
                    answer = self._call_gemini(prompt)
                    provider_used = "gemini"
                    model_used = self.gemini_model
                except Exception as gemini_error:
                    error_message = str(gemini_error)
                    answer = self._call_ollama(prompt)
                    provider_used = "ollama"
                    model_used = self.ollama_model

            return {
                "agent": self.agent_name,
                "provider": provider_used,
                "model": model_used,
                "question": question,
                "intent": intent,
                "answer": answer,
                "context": context,
                "agent_trace": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "reasoning_mode": "llm_router",
                    "data_source": self._data_source_for_intent(intent),
                    "odoo_enabled": False,
                    "fallback_error": error_message,
                    "next_stage": self._next_stage_for_intent(intent),
                },
                "disclaimer": (
                    "This is a simulated AI monitoring demo. "
                    "It is not a medical diagnosis and must be verified by clinical staff."
                ),
            }

        except Exception as e:
            fallback_language = self._detect_language(question)

            if fallback_language == "ar":
                answer = (
                    "تعذر على CareSense AI الوصول إلى نموذج الذكاء الاصطناعي حالياً. "
                    "تأكد أن Ollama يعمل محلياً وأن الموديل تم تحميله."
                )
            else:
                answer = (
                    "CareSense AI could not reach the AI model. "
                    "Please make sure Ollama is running locally and the model is installed."
                )

            return {
                "agent": self.agent_name,
                "provider": "none",
                "model": None,
                "question": question,
                "intent": intent,
                "answer": answer,
                "error": str(e),
                "context": context,
                "disclaimer": "This is a simulated AI monitoring demo and not a medical diagnosis.",
            }

    def _call_gemini(self, prompt: str) -> str:
        if not self.gemini_client:
            raise ValueError("Gemini client is not configured.")

        response = self.gemini_client.models.generate_content(
            model=self.gemini_model,
            contents=prompt,
        )

        return response.text or "No response returned from Gemini."

    def _call_ollama(self, prompt: str) -> str:
        url = f"{self.ollama_base_url}/api/generate"

        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.4,
                "top_p": 0.9,
            },
        }

        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()

        data = response.json()
        return data.get("response", "No response returned from Ollama.")

    def _detect_language(self, text: str) -> str:
        arabic_chars = len([c for c in text if "\u0600" <= c <= "\u06FF"])
        english_chars = len([c for c in text if c.isascii() and c.isalpha()])

        if arabic_chars > english_chars:
            return "ar"

        return "en"

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

    def _build_prompt(
        self,
        question: str,
        intent: str,
        context: Dict[str, Any],
        dashboard_payload: Dict[str, Any],
    ) -> str:
        language_rule = """
Language rule:
- Detect the language of the user's question.
- If the user writes in Arabic, reply ONLY in Arabic.
- If the user writes in English, reply ONLY in English.
- If the user mixes Arabic and English, reply mainly in the dominant language.
- Do not switch languages unless the user asks you to.
- Keep the tone professional, clear, natural, and conversational.
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
- general questions about your role and capabilities

Important safety rules:
- Do not claim to provide a real medical diagnosis.
- Do not invent patient data.
- Only use live monitoring data when the user's question is about the patient, monitoring, risk, fall, vitals, WiFi CSI, or emergency.
- For normal conversation, do not force a clinical report.
- Always remind the user that monitoring data is simulated when giving patient-related analysis.
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

Answer as CareSense AI.
Explain what you will be able to do once Odoo is connected.
Do not pretend that you already fetched real Odoo data.
Do not claim any action was completed unless the tool actually exists and was executed.
"""

        monitoring_context = self._safe_json(context)

        if intent == "wifi_csi":
            return f"""
{identity}
{language_rule}

User question:
{question}

The user is asking about WiFi CSI or signal analysis.

Use ONLY this live monitoring context:
{monitoring_context}

Answer with:
1. What the WiFi CSI signal indicates
2. How it relates to movement, breathing, or fall risk
3. Any limitation of the current simulated data

Do not invent values outside the provided context.
"""

        if intent == "emergency":
            return f"""
{identity}
{language_rule}

User question:
{question}

The user is asking about emergency action.

Use ONLY this live monitoring context:
{monitoring_context}

Give a clear triage-style response:
1. Current risk state
2. Evidence from the data
3. Whether immediate human verification is needed
4. Recommended next action

If risk_level is critical, fall_detected is true, breathing is abnormal, or heart rate is very high,
recommend immediate human verification.

Do not present this as a confirmed medical diagnosis.
"""

        if intent == "patient_monitoring":
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
Do not invent patient data.
"""

        return f"""
{identity}
{language_rule}

User question:
{question}

Answer naturally and helpfully as CareSense AI.
"""

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
            "patient_status_label": (patient.get("avatar", {}) or {}).get("status_label"),
            "system_recommended_action": ai_agent.get("recommended_action"),
            "system_summary": ai_agent.get("summary"),
        }

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
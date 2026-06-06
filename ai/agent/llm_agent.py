import os
import json
import requests
from datetime import datetime
from typing import Dict, Any, Optional

from dotenv import load_dotenv

from integrations.odoo.patients import OdooPatientService

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
        user_language = self._detect_language(question)
        context = self._extract_context(dashboard_payload)

        if intent == "odoo_patient_search":
            return self._run_odoo_patient_search(question, user_language)

        prompt = self._build_prompt(
            question=question,
            intent=intent,
            context=context,
            dashboard_payload=dashboard_payload,
            user_language=user_language,
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
            if user_language == "ar":
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

    def _run_odoo_patient_search(self, question: str, user_language: str) -> Dict[str, Any]:
        try:
            patient_query = self._extract_patient_search_query(question)

            if not patient_query:
                answer = (
                    "من فضلك اذكر اسم المريض أو رقم الهوية للبحث عنه في Odoo."
                    if user_language == "ar"
                    else "Please provide the patient name or ID number to search in Odoo."
                )

                return self._odoo_response(
                    question=question,
                    answer=answer,
                    tool="search_patient",
                    data=[],
                    error=None,
                )

            service = OdooPatientService()
            patients = service.search_patient(patient_query, limit=5)

            if not patients:
                answer = (
                    f"لم أجد أي مريض مطابق للبحث: {patient_query}"
                    if user_language == "ar"
                    else f"I could not find any patient matching: {patient_query}"
                )

                return self._odoo_response(
                    question=question,
                    answer=answer,
                    tool="search_patient",
                    data=[],
                    error=None,
                )

            summaries = []

            for patient in patients:
                patient_id = patient.get("id")
                full_summary = service.get_patient_full_summary(patient_id)
                summaries.append(full_summary or {"profile": patient, "latest_vitals": {}})

            answer = self._format_patient_full_summary(
                summaries=summaries,
                user_language=user_language,
            )

            return self._odoo_response(
                question=question,
                answer=answer,
                tool="get_patient_full_summary",
                data=summaries,
                error=None,
            )

        except Exception as e:
            answer = (
                f"حدث خطأ أثناء الاتصال بـ Odoo: {str(e)}"
                if user_language == "ar"
                else f"An error occurred while connecting to Odoo: {str(e)}"
            )

            return self._odoo_response(
                question=question,
                answer=answer,
                tool="get_patient_full_summary",
                data=[],
                error=str(e),
            )

    def _format_patient_full_summary(
            self,
            summaries: list,
            user_language: str,
    ) -> str:

        if user_language == "ar":
            lines = [f"وجدت {len(summaries)} نتيجة في Odoo:"]

            for item in summaries:
                profile = item.get("profile", {}) or {}
                vitals = item.get("latest_vitals", {}) or {}

                lines.append(
                    f"""
    الاسم: {profile.get("name") or "غير متوفر"}
    رقم السجل: {profile.get("id") or "غير متوفر"}
    المرجع: {profile.get("reference") or "غير متوفر"}
    رقم الهوية: {profile.get("ssn") or "غير متوفر"}
    الجوال: {profile.get("mobile") or "غير متوفر"}
    العمر: {profile.get("age") or "غير متوفر"}
    الجنس: {profile.get("sex") or "غير متوفر"}
    فصيلة الدم: {profile.get("blood_type") or "غير متوفرة"}
    حالة الملف: {profile.get("state") or "غير متوفرة"}
    حالة المريض: {profile.get("patient_condition") or "غير محددة"}

    آخر تشخيص: {profile.get("last_diagnosis") or "غير متوفر"}
    تاريخ التشخيص: {profile.get("diagnosis_date") or "غير متوفر"}

    معلومات حرجة: {profile.get("critical_info") or "لا توجد معلومات حرجة مسجلة"}
    معلومات عامة: {profile.get("general_info") or "لا توجد معلومات عامة مسجلة"}

    آخر علامات حيوية:
    - الحرارة: {vitals.get("last_temperature") or "غير متوفرة"}
    - النبض: {vitals.get("last_heart_rate") or "غير متوفر"}
    - الأكسجين: {vitals.get("last_oxygen_saturation") or "غير متوفر"}
    - معدل التنفس: {vitals.get("last_respiratory_rate") or "غير متوفر"}
    - الضغط الانقباضي: {vitals.get("last_systolic") or "غير متوفر"}
    - الضغط الانبساطي: {vitals.get("last_diastolic") or "غير متوفر"}
    - السكر: {vitals.get("last_glycemia") or "غير متوفر"}
    - الوزن: {vitals.get("last_weight") or "غير متوفر"}
    - الطول: {vitals.get("last_height") or "غير متوفر"}
    """.strip()
                )

            return "\n\n".join(lines)

        lines = [f"I found {len(summaries)} result(s) in Odoo:"]

        for item in summaries:
            profile = item.get("profile", {}) or {}
            vitals = item.get("latest_vitals", {}) or {}

            lines.append(
                f"""
    Name: {profile.get("name") or "N/A"}
    Record ID: {profile.get("id") or "N/A"}
    Reference: {profile.get("reference") or "N/A"}
    ID Number: {profile.get("ssn") or "N/A"}
    Mobile: {profile.get("mobile") or "N/A"}
    Age: {profile.get("age") or "N/A"}
    Gender: {profile.get("sex") or "N/A"}
    Blood Type: {profile.get("blood_type") or "N/A"}
    File Status: {profile.get("state") or "N/A"}
    Patient Condition: {profile.get("patient_condition") or "N/A"}

    Last Diagnosis: {profile.get("last_diagnosis") or "N/A"}
    Diagnosis Date: {profile.get("diagnosis_date") or "N/A"}

    Critical Info: {profile.get("critical_info") or "No critical information recorded"}
    General Info: {profile.get("general_info") or "No general information recorded"}

    Latest Vitals:
    - Temperature: {vitals.get("last_temperature") or "N/A"}
    - Heart Rate: {vitals.get("last_heart_rate") or "N/A"}
    - Oxygen Saturation: {vitals.get("last_oxygen_saturation") or "N/A"}
    - Respiratory Rate: {vitals.get("last_respiratory_rate") or "N/A"}
    - Systolic BP: {vitals.get("last_systolic") or "N/A"}
    - Diastolic BP: {vitals.get("last_diastolic") or "N/A"}
    - Glycemia: {vitals.get("last_glycemia") or "N/A"}
    - Weight: {vitals.get("last_weight") or "N/A"}
    - Height: {vitals.get("last_height") or "N/A"}
    """.strip()
            )

        return "\n\n".join(lines)

    def _odoo_response(
            self,
            question: str,
            answer: str,
            tool: str,
            data: Any,
            error: Optional[str],
    ) -> Dict[str, Any]:
        return {
            "agent": self.agent_name,
            "provider": "odoo_tool",
            "model": "sm.patient",
            "question": question,
            "intent": "odoo_patient_search",
            "answer": answer,
            "context": {
                "odoo_data": data,
            },
            "agent_trace": {
                "timestamp": datetime.utcnow().isoformat(),
                "reasoning_mode": "tool_call",
                "tool": tool,
                "data_source": "odoo_sm_patient",
                "odoo_enabled": True,
                "error": error,
            },
            "disclaimer": "Patient data retrieved from Odoo according to the configured Odoo user permissions.",
        }

    def _extract_patient_search_query(self, question: str) -> str:
        q = (question or "").strip()

        replacements = [
            "ابحث عن المريض",
            "ابحث عن مريض",
            "دور على المريض",
            "دور على مريض",
            "هات بيانات المريض",
            "بيانات المريض",
            "ملف المريض",
            "search patient",
            "find patient",
            "patient data",
            "patient profile",
        ]

        cleaned = q

        for item in replacements:
            cleaned = cleaned.replace(item, "")

        cleaned = cleaned.replace("؟", "").replace("?", "").strip()

        return cleaned

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
                "num_predict": 160,
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

        odoo_patient_keywords = [
            "ابحث عن المريض",
            "ابحث عن مريض",
            "دور على المريض",
            "دور على مريض",
            "بيانات المريض",
            "ملف المريض",
            "هات بيانات المريض",
            "search patient",
            "find patient",
            "patient data",
            "patient profile",
        ]

        general_keywords = [
            "hello", "hi", "hey", "good morning", "good evening",
            "who are you", "what are you", "can you speak", "speak arabic",
            "arabic", "english", "help", "مرحبا", "اهلا", "أهلا",
            "السلام", "تتكلم عربي", "عربي", "انت مين", "ما اسمك"
        ]

        odoo_keywords = [
            "odoo", "appointment", "book", "patient record", "medical record",
            "consultation", "invoice", "حجز", "موعد", "اودو", "أودو",
            "السجل الطبي", "استشارة"
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

        if any(k in q for k in odoo_patient_keywords):
            return "odoo_patient_search"

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
            user_language: str,
    ) -> str:
        if user_language == "ar":
            language_rule = """
STRICT LANGUAGE RULE:
- The user's question is Arabic.
- You MUST answer ONLY in Arabic.
- Do not answer in English.
- Do not include English section titles.
- Translate any section titles into Arabic.
- Use clear Modern Standard Arabic with simple medical wording.
"""
        else:
            language_rule = """
STRICT LANGUAGE RULE:
- The user's question is English.
- You MUST answer ONLY in English.
- Do not answer in Arabic.
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

        if user_language == "ar":
            patient_structure = """
التقييم السريري:
...

الأدلة الحالية:
...

تفسير مستوى الخطورة:
...

الإجراء الموصى به:
...

ملاحظة مهمة:
هذه بيانات مراقبة محاكاة وليست تشخيصاً طبياً.
"""
        else:
            patient_structure = """
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
Do not output clinical sections unless the user asks about the patient.
"""

        if intent == "odoo":
            return f"""
{identity}
{language_rule}

User question:
{question}

The user is asking about Odoo or healthcare workflow integration.

Current Odoo status:
- Odoo patient search tool is connected.
- Other future tools will include:
  1. get_patient_medical_record
  2. create_patient_alert
  3. book_appointment
  4. create_clinical_note

Answer as CareSense AI.
Explain what you can do now and what will be added later.
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

{patient_structure}

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
        if intent == "odoo_patient_search":
            return "odoo_sm_patient"
        if intent in ["patient_monitoring", "wifi_csi", "emergency"]:
            return "live_dashboard_payload"
        if intent == "odoo":
            return "planned_odoo_tools"
        return "conversation_only"

    def _next_stage_for_intent(self, intent: str) -> Optional[str]:
        if intent == "odoo_patient_search":
            return "get_patient_profile_and_vitals"
        if intent == "odoo":
            return "connect_more_odoo_tools"
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

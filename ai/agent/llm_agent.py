import os
import json
import requests
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

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

        self.current_patient_id: Optional[int] = None
        self.current_patient_summary: Optional[Dict[str, Any]] = None

    def run(self, question: str, dashboard_payload: Dict[str, Any]) -> Dict[str, Any]:
        question = (question or "").strip()
        user_language = self._detect_language(question)
        dashboard_context = self._extract_context(dashboard_payload)

        if not question:
            return self._normal_response(
                question=question,
                answer="من فضلك اكتب سؤالك." if user_language == "ar" else "Please write your question.",
                intent="empty",
                provider="system",
                model=None,
                context=dashboard_context,
                error=None,
            )

        try:
            decision = self._decide_next_action(
                question=question,
                user_language=user_language,
                dashboard_context=dashboard_context,
            )

            action = decision.get("action")
            patient_query = decision.get("patient_query")
            reason = decision.get("reason")

            if action == "search_patient":
                return self._tool_search_patient(
                    question=question,
                    patient_query=patient_query,
                    user_language=user_language,
                    reason=reason,
                )

            if action == "answer_current_patient":
                return self._answer_from_current_patient(
                    question=question,
                    user_language=user_language,
                    dashboard_context=dashboard_context,
                    reason=reason,
                )

            if action == "answer_dashboard":
                return self._answer_from_dashboard(
                    question=question,
                    user_language=user_language,
                    dashboard_context=dashboard_context,
                    reason=reason,
                )

            if action == "general_chat":
                return self._answer_general(
                    question=question,
                    user_language=user_language,
                    dashboard_context=dashboard_context,
                    reason=reason,
                )

            return self._normal_response(
                question=question,
                answer=(
                    "أحتاج أن تبحث عن المريض أولاً. مثال: ابحث عن المريض Ahmed Ali."
                    if user_language == "ar"
                    else "Please search for a patient first. Example: search patient Ahmed Ali."
                ),
                intent="need_patient_context",
                provider="system",
                model=None,
                context={
                    "current_patient": self.current_patient_summary,
                    "dashboard": dashboard_context,
                    "planner_decision": decision,
                },
                error=None,
            )

        except Exception as e:
            return self._normal_response(
                question=question,
                answer=(
                    f"حدث خطأ أثناء تشغيل CareSense AI: {str(e)}"
                    if user_language == "ar"
                    else f"CareSense AI error: {str(e)}"
                ),
                intent="error",
                provider="system",
                model=None,
                context={
                    "current_patient": self.current_patient_summary,
                    "dashboard": dashboard_context,
                },
                error=str(e),
            )

    # ------------------------------------------------------------------
    # LLM PLANNER
    # ------------------------------------------------------------------

    def _decide_next_action(
        self,
        question: str,
        user_language: str,
        dashboard_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        current_patient_brief = None

        if self.current_patient_summary:
            profile = self.current_patient_summary.get("profile", {}) or {}
            current_patient_brief = {
                "id": profile.get("id"),
                "name": profile.get("name"),
                "reference": profile.get("reference"),
                "ssn": profile.get("ssn"),
            }

        prompt = f"""
You are the planner for CareSense AI.

You must decide what the system should do next.

Available actions:
1. search_patient
   Use this when the user is asking to find, open, load, select, or start working with a patient.
   Extract the patient name, ID, SSN, or reference into patient_query.

2. answer_current_patient
   Use this when the user asks anything about the currently selected patient.
   This includes profile, vitals, diagnosis, treatment, clinical interpretation, appointments, or any follow-up using pronouns like he/she/his/her/هو/هي/حالته.

3. answer_dashboard
   Use this when the user asks about live monitoring, WiFi sensing, fall detection, room status, movement, breathing, or the simulated dashboard.

4. general_chat
   Use this for normal conversation not requiring Odoo patient data or dashboard data.

Important:
- Do not answer the user.
- Return valid JSON only.
- Do not use markdown.
- If there is no current patient and the user asks about a patient without giving a searchable patient identity, return action "need_patient".
- patient_query must be null unless action is search_patient.
- You are multilingual. Understand Arabic, English, and mixed Arabic-English.

Current selected patient:
{self._safe_json(current_patient_brief)}

Live dashboard context:
{self._safe_json(dashboard_context)}

User message:
{question}

Return JSON with exactly this shape:
{{
  "action": "search_patient | answer_current_patient | answer_dashboard | general_chat | need_patient",
  "patient_query": null,
  "reason": "short reason"
}}
"""

        raw_answer, _, _, _ = self._call_llm(prompt)
        decision = self._parse_json(raw_answer)

        if not isinstance(decision, dict):
            return {
                "action": "general_chat",
                "patient_query": None,
                "reason": "Planner returned invalid JSON.",
            }

        action = decision.get("action")
        if action not in [
            "search_patient",
            "answer_current_patient",
            "answer_dashboard",
            "general_chat",
            "need_patient",
        ]:
            action = "general_chat"

        return {
            "action": action,
            "patient_query": decision.get("patient_query"),
            "reason": decision.get("reason"),
        }

    # ------------------------------------------------------------------
    # ODOO TOOLS
    # ------------------------------------------------------------------

    def _tool_search_patient(
        self,
        question: str,
        patient_query: Optional[str],
        user_language: str,
        reason: Optional[str],
    ) -> Dict[str, Any]:
        patient_query = (patient_query or "").strip()

        if not patient_query:
            return self._normal_response(
                question=question,
                answer=(
                    "من فضلك اذكر اسم المريض أو رقم الهوية أو المرجع."
                    if user_language == "ar"
                    else "Please provide the patient name, ID number, or reference."
                ),
                intent="search_patient",
                provider="odoo_tool",
                model="sm.patient",
                context={
                    "tool": "search_patient",
                    "planner_reason": reason,
                    "odoo_data": [],
                },
                error=None,
            )

        try:
            service = OdooPatientService()
            patients = service.search_patient(patient_query, limit=5)

            if not patients:
                return self._normal_response(
                    question=question,
                    answer=(
                        f"لم أجد مريضاً مطابقاً لـ: {patient_query}"
                        if user_language == "ar"
                        else f"I could not find a patient matching: {patient_query}"
                    ),
                    intent="search_patient",
                    provider="odoo_tool",
                    model="sm.patient",
                    context={
                        "tool": "search_patient",
                        "planner_reason": reason,
                        "patient_query": patient_query,
                        "odoo_data": [],
                    },
                    error=None,
                )

            summaries = []

            for patient in patients:
                patient_id = patient.get("id")
                full_summary = service.get_patient_context_for_chat(patient_id)
                summaries.append(full_summary or {"profile": patient, "latest_vitals": {}})

            if len(summaries) == 1:
                self.current_patient_summary = summaries[0]
                self.current_patient_id = (
                    summaries[0].get("profile", {}) or {}
                ).get("id")

                answer = self._format_loaded_patient_message(
                    patient_summary=summaries[0],
                    user_language=user_language,
                )

                return self._normal_response(
                    question=question,
                    answer=answer,
                    intent="search_patient",
                    provider="odoo_tool",
                    model="sm.patient",
                    context={
                        "tool": "search_patient",
                        "planner_reason": reason,
                        "patient_query": patient_query,
                        "current_patient": self.current_patient_summary,
                        "odoo_data": summaries,
                    },
                    error=None,
                )

            return self._normal_response(
                question=question,
                answer=self._format_patient_selection(summaries, user_language),
                intent="select_patient",
                provider="odoo_tool",
                model="sm.patient",
                context={
                    "tool": "search_patient",
                    "planner_reason": reason,
                    "patient_query": patient_query,
                    "odoo_data": summaries,
                },
                error=None,
            )

        except Exception as e:
            return self._normal_response(
                question=question,
                answer=(
                    f"حدث خطأ أثناء الاتصال بـ Odoo: {str(e)}"
                    if user_language == "ar"
                    else f"An error occurred while connecting to Odoo: {str(e)}"
                ),
                intent="search_patient_error",
                provider="odoo_tool",
                model="sm.patient",
                context={
                    "tool": "search_patient",
                    "planner_reason": reason,
                    "patient_query": patient_query,
                },
                error=str(e),
            )

    def _answer_from_current_patient(
        self,
        question: str,
        user_language: str,
        dashboard_context: Dict[str, Any],
        reason: Optional[str],
    ) -> Dict[str, Any]:
        if not self.current_patient_summary:
            return self._normal_response(
                question=question,
                answer=(
                    "لا يوجد مريض محدد حالياً. ابحث عن المريض أولاً، وبعدها اسألني عنه بأي صيغة."
                    if user_language == "ar"
                    else "There is no selected patient yet. Search for a patient first, then ask me about them freely."
                ),
                intent="need_patient_context",
                provider="system",
                model=None,
                context={
                    "planner_reason": reason,
                    "dashboard": dashboard_context,
                },
                error=None,
            )

        prompt = self._build_patient_answer_prompt(
            question=question,
            patient_summary=self.current_patient_summary,
            dashboard_context=dashboard_context,
            user_language=user_language,
        )

        answer, provider_used, model_used, error_message = self._call_llm(prompt)

        return self._normal_response(
            question=question,
            answer=answer,
            intent="answer_current_patient",
            provider=provider_used,
            model=model_used,
            context={
                "planner_reason": reason,
                "current_patient": self.current_patient_summary,
                "dashboard": dashboard_context,
            },
            error=error_message,
        )

    # ------------------------------------------------------------------
    # ANSWER PROMPTS
    # ------------------------------------------------------------------

    def _build_patient_answer_prompt(
        self,
        question: str,
        patient_summary: Dict[str, Any],
        dashboard_context: Dict[str, Any],
        user_language: str,
    ) -> str:
        language_rule = (
            "Answer only in Arabic. Use clear simple Arabic."
            if user_language == "ar"
            else "Answer only in English. Use clear simple clinical language."
        )

        return f"""
You are CareSense AI, a smart medical assistant connected to Odoo.

You are answering a free chat question about the currently selected patient.

Rules:
- Use ONLY the Odoo patient data provided.
- Do not invent missing values.
- If the requested information is not available, say it is not available in the current Odoo record.
- You can interpret vitals cautiously, but you must not give a confirmed diagnosis.
- If the user asks whether the patient needs a doctor or appointment, answer based on the available data and recommend clinical review when appropriate.
- Do not expose raw JSON unless the user asks for raw data.
- Be natural like a chat assistant, not a fixed report.
- {language_rule}

User question:
{question}

Selected Odoo patient data:
{self._safe_json(patient_summary)}

Live dashboard context, if relevant:
{self._safe_json(dashboard_context)}
"""

    def _answer_from_dashboard(
        self,
        question: str,
        user_language: str,
        dashboard_context: Dict[str, Any],
        reason: Optional[str],
    ) -> Dict[str, Any]:
        language_rule = (
            "أجب باللغة العربية فقط."
            if user_language == "ar"
            else "Answer in English only."
        )

        prompt = f"""
You are CareSense AI.

Answer the user's question using only the live dashboard context below.

Rules:
- The dashboard data is simulated WiFi CSI monitoring data.
- Do not invent values.
- Do not provide a confirmed medical diagnosis.
- Be concise and natural.
- {language_rule}

User question:
{question}

Dashboard context:
{self._safe_json(dashboard_context)}
"""

        answer, provider_used, model_used, error_message = self._call_llm(prompt)

        return self._normal_response(
            question=question,
            answer=answer,
            intent="answer_dashboard",
            provider=provider_used,
            model=model_used,
            context={
                "planner_reason": reason,
                "dashboard": dashboard_context,
            },
            error=error_message,
        )

    def _answer_general(
        self,
        question: str,
        user_language: str,
        dashboard_context: Dict[str, Any],
        reason: Optional[str],
    ) -> Dict[str, Any]:
        current_patient_brief = None

        if self.current_patient_summary:
            profile = self.current_patient_summary.get("profile", {}) or {}
            current_patient_brief = {
                "id": profile.get("id"),
                "name": profile.get("name"),
                "reference": profile.get("reference"),
            }

        language_rule = (
            "أجب باللغة العربية فقط."
            if user_language == "ar"
            else "Answer in English only."
        )

        prompt = f"""
You are CareSense AI, a conversational medical monitoring assistant.

Rules:
- Be natural and helpful.
- Do not invent patient data.
- If the user asks about a patient, explain that a patient can be loaded from Odoo first.
- {language_rule}

Current selected patient brief:
{self._safe_json(current_patient_brief)}

User question:
{question}
"""

        answer, provider_used, model_used, error_message = self._call_llm(prompt)

        return self._normal_response(
            question=question,
            answer=answer,
            intent="general_chat",
            provider=provider_used,
            model=model_used,
            context={
                "planner_reason": reason,
                "current_patient": current_patient_brief,
                "dashboard": dashboard_context,
            },
            error=error_message,
        )

    # ------------------------------------------------------------------
    # FORMATTERS
    # ------------------------------------------------------------------

    def _format_loaded_patient_message(
        self,
        patient_summary: Dict[str, Any],
        user_language: str,
    ) -> str:
        profile = patient_summary.get("profile", {}) or {}
        vitals = patient_summary.get("latest_vitals", {}) or {}

        if user_language == "ar":
            return (
                f"تم تحميل المريض كمريض حالي.\n\n"
                f"الاسم: {profile.get('name') or 'غير متوفر'}\n"
                f"رقم السجل: {profile.get('id') or 'غير متوفر'}\n"
                f"المرجع: {profile.get('reference') or 'غير متوفر'}\n"
                f"العمر: {profile.get('age') or 'غير متوفر'}\n"
                f"الحالة: {profile.get('patient_condition') or 'غير محددة'}\n\n"
                f"يمكنك الآن أن تسألني عنه بأي صيغة، مثل: هل ضغطه مقلق؟ ما آخر تشخيص؟ هل يحتاج موعد؟"
            )

        return (
            f"Patient loaded as the current patient.\n\n"
            f"Name: {profile.get('name') or 'N/A'}\n"
            f"Record ID: {profile.get('id') or 'N/A'}\n"
            f"Reference: {profile.get('reference') or 'N/A'}\n"
            f"Age: {profile.get('age') or 'N/A'}\n"
            f"Condition: {profile.get('patient_condition') or 'N/A'}\n\n"
            f"You can now ask about this patient freely."
        )

    def _format_patient_selection(
        self,
        summaries: List[Dict[str, Any]],
        user_language: str,
    ) -> str:
        if user_language == "ar":
            lines = ["وجدت أكثر من مريض مطابق. من فضلك حدد المريض برقم السجل:"]
            for item in summaries:
                profile = item.get("profile", {}) or {}
                lines.append(
                    f"- رقم السجل: {profile.get('id')}, "
                    f"الاسم: {profile.get('name') or 'غير متوفر'}, "
                    f"المرجع: {profile.get('reference') or 'غير متوفر'}, "
                    f"رقم الهوية: {profile.get('ssn') or 'غير متوفر'}"
                )
            return "\n".join(lines)

        lines = ["I found more than one matching patient. Please specify the record ID:"]
        for item in summaries:
            profile = item.get("profile", {}) or {}
            lines.append(
                f"- Record ID: {profile.get('id')}, "
                f"Name: {profile.get('name') or 'N/A'}, "
                f"Reference: {profile.get('reference') or 'N/A'}, "
                f"ID Number: {profile.get('ssn') or 'N/A'}"
            )
        return "\n".join(lines)

    def _normal_response(
        self,
        question: str,
        answer: str,
        intent: str,
        provider: str,
        model: Optional[str],
        context: Dict[str, Any],
        error: Optional[str],
    ) -> Dict[str, Any]:
        return {
            "agent": self.agent_name,
            "provider": provider,
            "model": model,
            "question": question,
            "intent": intent,
            "answer": answer,
            "context": context,
            "agent_trace": {
                "timestamp": datetime.utcnow().isoformat(),
                "reasoning_mode": "llm_tool_planner",
                "odoo_enabled": True,
                "current_patient_id": self.current_patient_id,
                "error": error,
            },
            "disclaimer": (
                "Patient data is retrieved from Odoo according to configured permissions. "
                "This is not a confirmed medical diagnosis."
            ),
        }

    # ------------------------------------------------------------------
    # LLM CALLS
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> Tuple[str, str, Optional[str], Optional[str]]:
        provider_used = None
        model_used = None
        error_message = None

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

        return answer, provider_used, model_used, error_message

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
                "temperature": 0.2,
                "top_p": 0.9,
                "num_predict": 500,
            },
        }

        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()

        data = response.json()
        return data.get("response", "No response returned from Ollama.")

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _parse_json(self, text: str) -> Dict[str, Any]:
        if not text:
            return {}

        text = text.strip()

        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "").strip()

        try:
            return json.loads(text)
        except Exception:
            pass

        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                return {}

        return {}

    def _detect_language(self, text: str) -> str:
        arabic_chars = len([c for c in text if "\u0600" <= c <= "\u06FF"])
        english_chars = len([c for c in text if c.isascii() and c.isalpha()])

        if arabic_chars > english_chars:
            return "ar"

        return "en"

    def _extract_context(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        payload = payload or {}

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

    def _safe_json(self, data: Any) -> str:
        try:
            return json.dumps(data, indent=2, ensure_ascii=False, default=str)
        except Exception:
            return str(data)


_agent_instance = CareSenseLLMAgent()


def generate_llm_chat_response(
    question: str,
    dashboard_payload: dict,
    session_id: str,
) -> dict:

    return _agent_instance.run(
        question=question,
        dashboard_payload=dashboard_payload,
        session_id=session_id,
    )
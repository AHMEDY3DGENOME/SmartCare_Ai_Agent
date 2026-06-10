import os
import re
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from integrations.odoo.patients import OdooPatientService

load_dotenv()


SESSIONS: Dict[str, Dict[str, Any]] = {}


# ======================================================================
# Basic Helpers
# ======================================================================

def _safe_json(data: Any) -> str:
    try:
        return json.dumps(
            data,
            ensure_ascii=False,
            default=str,
            indent=2,
        )
    except Exception:
        return str(data)


def _detect_language(text: str) -> str:
    arabic_chars = len([c for c in text if "\u0600" <= c <= "\u06FF"])
    english_chars = len([c for c in text if c.isascii() and c.isalpha()])

    return "ar" if arabic_chars > english_chars else "en"


def _get_session(session_id: str) -> Dict[str, Any]:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "current_patient": None,
            "current_patient_id": None,
            "history": [],
        }

    return SESSIONS[session_id]


def _append_history(
    session: Dict[str, Any],
    question: str,
    answer: str,
) -> None:
    history = session.setdefault("history", [])

    history.append({
        "user": question,
        "assistant": answer,
        "timestamp": datetime.utcnow().isoformat(),
    })

    session["history"] = history[-10:]


def _patient_identity(
    patient_context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not patient_context:
        return {}

    return (
        patient_context.get("patient_identity")
        or patient_context.get("profile")
        or {}
    )


def _get_patient_context(
    service: OdooPatientService,
    patient_id: int,
) -> Optional[Dict[str, Any]]:
    if hasattr(service, "get_patient_context_for_chat"):
        return service.get_patient_context_for_chat(patient_id)

    return service.get_patient_full_summary(patient_id)


# ======================================================================
# Deterministic Patient Router
# ======================================================================

def _normalize_text(text: str) -> str:
    text = (text or "").strip()
    text = text.replace("؟", "?")
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_patient_load_query(question: str) -> Optional[str]:
    q = _normalize_text(question)
    q_lower = q.lower()

    prefixes = [
        "ابحث عن المريض",
        "ابحث عن مريض",
        "دور على المريض",
        "دور على مريض",
        "افتح المريض",
        "افتح مريض",
        "حمل المريض",
        "حمّل المريض",
        "اعرض المريض",
        "اعرض بيانات المريض",
        "هات بيانات المريض",
        "بيانات المريض",
        "ملف المريض",
        "search patient",
        "find patient",
        "load patient",
        "open patient",
        "show patient",
        "patient profile",
        "patient data",
        "patient",
    ]

    for prefix in prefixes:
        if q_lower.startswith(prefix.lower()):
            query = q[len(prefix):].strip(" .,:;؟?")
            return query or None

    return None


def _extract_patient_id_selection(question: str) -> Optional[int]:
    q = _normalize_text(question).lower()

    patterns = [
        r"(?:اختار|اختر|حدد|رقم السجل|رقم|سجل)\s+(\d+)",
        r"(?:select|choose|pick|record id|patient id|id)\s+(\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, q, flags=re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                return None

    return None


def _load_patient_directly(
    session_id: str,
    session: Dict[str, Any],
    question: str,
    patient_query: str,
    user_language: str,
) -> Dict[str, Any]:
    service = OdooPatientService()

    try:
        patients = service.search_patient(patient_query, limit=5)

        if not patients:
            answer = (
                f"لم أجد مريضاً مطابقاً لـ: {patient_query}"
                if user_language == "ar"
                else f"I could not find a patient matching: {patient_query}"
            )

            _append_history(session, question, answer)

            return _build_response(
                session_id=session_id,
                question=question,
                answer=answer,
                intent="patient_not_found",
                provider="odoo_direct",
                model=None,
                context={
                    "patient_query": patient_query,
                    "odoo_data": [],
                    "current_patient_id": session.get("current_patient_id"),
                    "current_patient": session.get("current_patient"),
                },
                trace={
                    "framework": "deterministic_odoo_router",
                    "reasoning_mode": "direct_patient_search",
                    "odoo_enabled": True,
                    "error": None,
                },
            )

        summaries = []

        for patient in patients:
            patient_id = patient.get("id")
            patient_context = _get_patient_context(service, patient_id)
            summaries.append(patient_context or {"patient_identity": patient})

        if len(summaries) == 1:
            selected_patient = summaries[0]
            identity = _patient_identity(selected_patient)

            patient_id = identity.get("id") or patients[0].get("id")

            session["current_patient_id"] = patient_id
            session["current_patient"] = selected_patient

            patient_name = identity.get("name") or patients[0].get("name") or patient_query

            answer = (
                f"تم تحميل المريض {patient_name} بنجاح كمريض حالي. يمكنك الآن سؤالي عنه بحرية."
                if user_language == "ar"
                else f"Patient {patient_name} has been loaded successfully as the current patient. You can now ask about this patient freely."
            )

            _append_history(session, question, answer)

            return _build_response(
                session_id=session_id,
                question=question,
                answer=answer,
                intent="patient_loaded",
                provider="odoo_direct",
                model="sm.patient",
                context={
                    "patient_query": patient_query,
                    "current_patient_id": session.get("current_patient_id"),
                    "current_patient": session.get("current_patient"),
                    "odoo_data": summaries,
                },
                trace={
                    "framework": "deterministic_odoo_router",
                    "reasoning_mode": "direct_patient_load",
                    "odoo_enabled": True,
                    "error": None,
                },
            )

        answer = _format_patient_selection(
            summaries=summaries,
            user_language=user_language,
        )

        _append_history(session, question, answer)

        return _build_response(
            session_id=session_id,
            question=question,
            answer=answer,
            intent="multiple_patient_matches",
            provider="odoo_direct",
            model="sm.patient",
            context={
                "patient_query": patient_query,
                "odoo_data": summaries,
                "current_patient_id": session.get("current_patient_id"),
                "current_patient": session.get("current_patient"),
            },
            trace={
                "framework": "deterministic_odoo_router",
                "reasoning_mode": "direct_patient_search_multiple_matches",
                "odoo_enabled": True,
                "error": None,
            },
        )

    except Exception as e:
        answer = (
            f"حدث خطأ أثناء قراءة بيانات المريض من Odoo: {str(e)}"
            if user_language == "ar"
            else f"An error occurred while reading patient data from Odoo: {str(e)}"
        )

        _append_history(session, question, answer)

        return _build_response(
            session_id=session_id,
            question=question,
            answer=answer,
            intent="patient_load_error",
            provider="odoo_direct",
            model="sm.patient",
            context={
                "patient_query": patient_query,
                "current_patient_id": session.get("current_patient_id"),
                "current_patient": session.get("current_patient"),
            },
            trace={
                "framework": "deterministic_odoo_router",
                "reasoning_mode": "direct_patient_load_error",
                "odoo_enabled": True,
                "error": str(e),
            },
        )


def _select_patient_directly(
    session_id: str,
    session: Dict[str, Any],
    question: str,
    patient_id: int,
    user_language: str,
) -> Dict[str, Any]:
    service = OdooPatientService()

    try:
        patient_context = _get_patient_context(service, patient_id)

        if not patient_context:
            answer = (
                f"لم أجد مريضاً برقم السجل: {patient_id}"
                if user_language == "ar"
                else f"I could not find a patient with record ID: {patient_id}"
            )

            _append_history(session, question, answer)

            return _build_response(
                session_id=session_id,
                question=question,
                answer=answer,
                intent="patient_not_found_by_id",
                provider="odoo_direct",
                model="sm.patient",
                context={
                    "patient_id": patient_id,
                    "current_patient_id": session.get("current_patient_id"),
                    "current_patient": session.get("current_patient"),
                },
                trace={
                    "framework": "deterministic_odoo_router",
                    "reasoning_mode": "direct_patient_select_by_id",
                    "odoo_enabled": True,
                    "error": None,
                },
            )

        identity = _patient_identity(patient_context)

        session["current_patient_id"] = patient_id
        session["current_patient"] = patient_context

        patient_name = identity.get("name") or str(patient_id)

        answer = (
            f"تم اختيار المريض {patient_name} كمريض حالي. يمكنك الآن سؤالي عنه بحرية."
            if user_language == "ar"
            else f"Patient {patient_name} has been selected as the current patient. You can now ask about this patient freely."
        )

        _append_history(session, question, answer)

        return _build_response(
            session_id=session_id,
            question=question,
            answer=answer,
            intent="patient_selected",
            provider="odoo_direct",
            model="sm.patient",
            context={
                "patient_id": patient_id,
                "current_patient_id": session.get("current_patient_id"),
                "current_patient": session.get("current_patient"),
            },
            trace={
                "framework": "deterministic_odoo_router",
                "reasoning_mode": "direct_patient_select_by_id",
                "odoo_enabled": True,
                "error": None,
            },
        )

    except Exception as e:
        answer = (
            f"حدث خطأ أثناء اختيار المريض من Odoo: {str(e)}"
            if user_language == "ar"
            else f"An error occurred while selecting the patient from Odoo: {str(e)}"
        )

        _append_history(session, question, answer)

        return _build_response(
            session_id=session_id,
            question=question,
            answer=answer,
            intent="patient_select_error",
            provider="odoo_direct",
            model="sm.patient",
            context={
                "patient_id": patient_id,
                "current_patient_id": session.get("current_patient_id"),
                "current_patient": session.get("current_patient"),
            },
            trace={
                "framework": "deterministic_odoo_router",
                "reasoning_mode": "direct_patient_select_error",
                "odoo_enabled": True,
                "error": str(e),
            },
        )


# ======================================================================
# Direct LLM answer from loaded patient
# ======================================================================

def _value_or_na(value: Any, user_language: str) -> str:
    if value is None or value == "":
        return "غير متوفر" if user_language == "ar" else "Not available"
    return str(value)


def _is_yes(value: Any) -> bool:
    return str(value).strip().lower() in ("yes", "true", "1")


def _render_allergies(allergies: Dict[str, Any], user_language: str) -> str:
    allergies = allergies or {}
    drug = allergies.get("drug_allergy") or {}
    food = allergies.get("food_allergy") or {}
    other = allergies.get("other_allergy") or {}

    lines: List[str] = []

    if user_language == "ar":
        if _is_yes(drug.get("has")):
            lines.append(f"- حساسية دواء: نعم ({_value_or_na(drug.get('details'), user_language)})")
        if _is_yes(food.get("has")):
            lines.append(f"- حساسية طعام: نعم ({_value_or_na(food.get('details'), user_language)})")
        if _is_yes(other.get("has")):
            lines.append(f"- حساسية أخرى: نعم ({_value_or_na(other.get('details'), user_language)})")
        if not lines:
            lines.append("- لا توجد حساسية مسجلة في ملف Odoo.")
        return "\n".join(lines)

    if _is_yes(drug.get("has")):
        lines.append(f"- Drug allergy: Yes ({_value_or_na(drug.get('details'), user_language)})")
    if _is_yes(food.get("has")):
        lines.append(f"- Food allergy: Yes ({_value_or_na(food.get('details'), user_language)})")
    if _is_yes(other.get("has")):
        lines.append(f"- Other allergy: Yes ({_value_or_na(other.get('details'), user_language)})")
    if not lines:
        lines.append("- No allergy recorded in the Odoo file.")
    return "\n".join(lines)


def _render_chronic_conditions(chronic: Dict[str, Any], user_language: str) -> str:
    chronic = chronic or {}

    labels = {
        "ar": {
            "diabetes": "السكري",
            "hypertension": "ارتفاع ضغط الدم",
            "heart_diseases": "أمراض القلب",
            "cancer": "السرطان",
            "chronic_kidney_disease": "مرض الكلى المزمن",
            "chronic_respiratory_diseases": "أمراض الجهاز التنفسي المزمنة",
            "arthritis": "التهاب المفاصل",
        },
        "en": {
            "diabetes": "Diabetes",
            "hypertension": "Hypertension",
            "heart_diseases": "Heart diseases",
            "cancer": "Cancer",
            "chronic_kidney_disease": "Chronic kidney disease",
            "chronic_respiratory_diseases": "Chronic respiratory diseases",
            "arthritis": "Arthritis",
        },
    }

    lang_labels = labels["ar"] if user_language == "ar" else labels["en"]

    present = [label for key, label in lang_labels.items() if chronic.get(key)]

    if chronic.get("other_chronic_condition"):
        specific = chronic.get("specific_chronic_condition")
        if specific:
            present.append(str(specific))
        else:
            present.append("مرض مزمن آخر" if user_language == "ar" else "Other chronic condition")

    if not present:
        return (
            "- لا توجد أمراض مزمنة مسجلة في ملف Odoo."
            if user_language == "ar"
            else "- No chronic conditions recorded in the Odoo file."
        )

    return "\n".join(f"- {item}" for item in present)


def _render_medications(medications: List[Dict[str, Any]], user_language: str) -> str:
    medications = medications or []

    if not medications:
        return (
            "- لا توجد أدوية مسجلة في ملف Odoo."
            if user_language == "ar"
            else "- No medications recorded in the Odoo file."
        )

    lines: List[str] = []

    for medication in medications:
        name = medication.get("name") or ("دواء غير معروف" if user_language == "ar" else "Unknown medicine")
        segments = [f"- {name}"]

        if user_language == "ar":
            if medication.get("route"):
                segments.append(f"الطريق: {medication['route']}")
            if medication.get("dose"):
                dose = str(medication["dose"])
                if medication.get("unit"):
                    dose += f" {medication['unit']}"
                segments.append(f"الجرعة: {dose}")
            if medication.get("frequency_type"):
                freq = str(medication["frequency_type"])
                if medication.get("times_per_day"):
                    freq += f" ({medication['times_per_day']} مرات/يوم)"
                segments.append(f"التكرار: {freq}")
            if medication.get("duration"):
                period = medication.get("duration_period") or ""
                segments.append(f"المدة: {medication['duration']} {period}".strip())
            if medication.get("start_date") or medication.get("end_date"):
                segments.append(
                    f"من {medication.get('start_date') or '—'} إلى {medication.get('end_date') or '—'}"
                )
            if medication.get("indication"):
                segments.append(f"دواعي الاستعمال: {medication['indication']}")
            if medication.get("state"):
                segments.append(f"الحالة: {medication['state']}")
        else:
            if medication.get("route"):
                segments.append(f"Route: {medication['route']}")
            if medication.get("dose"):
                dose = str(medication["dose"])
                if medication.get("unit"):
                    dose += f" {medication['unit']}"
                segments.append(f"Dose: {dose}")
            if medication.get("frequency_type"):
                freq = str(medication["frequency_type"])
                if medication.get("times_per_day"):
                    freq += f" ({medication['times_per_day']}/day)"
                segments.append(f"Frequency: {freq}")
            if medication.get("duration"):
                period = medication.get("duration_period") or ""
                segments.append(f"Duration: {medication['duration']} {period}".strip())
            if medication.get("start_date") or medication.get("end_date"):
                segments.append(
                    f"From {medication.get('start_date') or '—'} to {medication.get('end_date') or '—'}"
                )
            if medication.get("indication"):
                segments.append(f"Indication: {medication['indication']}")
            if medication.get("state"):
                segments.append(f"State: {medication['state']}")

        lines.append(" | ".join(segments))

    return "\n".join(lines)


def _build_patient_readable_context(
    patient_context: Dict[str, Any],
    user_language: str,
) -> str:
    identity = patient_context.get("patient_identity") or patient_context.get("profile") or {}
    vitals = patient_context.get("latest_vitals") or {}
    clinical = patient_context.get("clinical_context") or {}
    emergency = patient_context.get("emergency_contact") or {}

    bp = vitals.get("blood_pressure") or {}
    temp = vitals.get("temperature") or {}
    hr = vitals.get("heart_rate") or {}
    oxygen = vitals.get("oxygen_saturation") or {}
    rr = vitals.get("respiratory_rate") or {}
    glycemia = vitals.get("glycemia") or {}
    weight = vitals.get("weight") or {}
    height = vitals.get("height") or {}

    last_diagnosis = clinical.get("last_diagnosis") or {}
    last_treatment_plan = clinical.get("last_treatment_plan") or {}
    last_visit_summary = clinical.get("last_visit_summary") or {}

    allergies = patient_context.get("allergies") or {}
    chronic = patient_context.get("chronic_conditions") or {}

    medications = patient_context.get("active_medications")
    if not medications:
        medications = patient_context.get("medications") or []

    allergies_text = _render_allergies(allergies, user_language)
    chronic_text = _render_chronic_conditions(chronic, user_language)
    medications_text = _render_medications(medications, user_language)

    if user_language == "ar":
        return f"""
سياق المريض الحالي من Odoo:

بيانات الهوية:
- الاسم: {_value_or_na(identity.get("name"), user_language)}
- رقم السجل: {_value_or_na(identity.get("id"), user_language)}
- المرجع: {_value_or_na(identity.get("reference"), user_language)}
- رقم الهوية: {_value_or_na(identity.get("ssn"), user_language)}
- العمر: {_value_or_na(identity.get("age"), user_language)}
- الجنس: {_value_or_na(identity.get("sex"), user_language)}
- فصيلة الدم: {_value_or_na(identity.get("blood_type"), user_language)}
- تاريخ الميلاد: {_value_or_na(identity.get("dob"), user_language)}
- الجنسية: {_value_or_na(identity.get("nationality"), user_language)}
- الحالة الاجتماعية: {_value_or_na(identity.get("marital_status"), user_language)}
- حالة الملف: {_value_or_na(identity.get("state"), user_language)}
- حالة المريض: {_value_or_na(identity.get("patient_condition"), user_language)}
- منوم حالياً: {_value_or_na(identity.get("patient_admitted"), user_language)}

آخر العلامات الحيوية:
- درجة الحرارة: {_value_or_na(temp.get("value"), user_language)} °C، التاريخ: {_value_or_na(temp.get("date"), user_language)}
- ضغط الدم: {_value_or_na(bp.get("systolic"), user_language)}/{_value_or_na(bp.get("diastolic"), user_language)} mmHg، التاريخ: {_value_or_na(bp.get("systolic_date") or bp.get("diastolic_date"), user_language)}
- الضغط الانقباضي Systolic: {_value_or_na(bp.get("systolic"), user_language)} mmHg
- الضغط الانبساطي Diastolic: {_value_or_na(bp.get("diastolic"), user_language)} mmHg
- معدل نبض القلب: {_value_or_na(hr.get("value"), user_language)} bpm، التاريخ: {_value_or_na(hr.get("date"), user_language)}
- نسبة الأكسجين: {_value_or_na(oxygen.get("value"), user_language)}%، التاريخ: {_value_or_na(oxygen.get("date"), user_language)}
- معدل التنفس: {_value_or_na(rr.get("value"), user_language)} /min، التاريخ: {_value_or_na(rr.get("date"), user_language)}
- السكر: {_value_or_na(glycemia.get("value"), user_language)}، التاريخ: {_value_or_na(glycemia.get("date"), user_language)}
- الوزن: {_value_or_na(weight.get("value"), user_language)} kg، التاريخ: {_value_or_na(weight.get("date"), user_language)}
- الطول: {_value_or_na(height.get("value"), user_language)} cm، التاريخ: {_value_or_na(height.get("date"), user_language)}

السياق السريري:
- معلومات حرجة: {_value_or_na(clinical.get("critical_info"), user_language)}
- معلومات عامة: {_value_or_na(clinical.get("general_info"), user_language)}
- آخر تشخيص: {_value_or_na(last_diagnosis.get("value"), user_language)}، التاريخ: {_value_or_na(last_diagnosis.get("date"), user_language)}
- آخر خطة علاجية: {_value_or_na(last_treatment_plan.get("value"), user_language)}، التاريخ: {_value_or_na(last_treatment_plan.get("date"), user_language)}
- آخر ملخص زيارة: {_value_or_na(last_visit_summary.get("value"), user_language)}، التاريخ: {_value_or_na(last_visit_summary.get("date"), user_language)}

الحساسية:
{allergies_text}

الأمراض المزمنة:
{chronic_text}

الأدوية الحالية (Active):
{medications_text}

بيانات الطوارئ:
- الاسم: {_value_or_na(emergency.get("name"), user_language)}
- الجوال: {_value_or_na(emergency.get("mobile"), user_language)}
- صلة القرابة: {_value_or_na(emergency.get("relation"), user_language)}

ملاحظات تفسيرية:
- عندما يسأل المستخدم عن "الضغط"، المقصود هو ضغط الدم: الانقباضي والانبساطي.
- عندما يسأل عن "النبض"، المقصود معدل نبض القلب.
- عندما يسأل عن "الأكسجين"، المقصود نسبة تشبع الأكسجين.
- عندما يسأل عن "السكر"، المقصود glycemia.
- عندما يسأل عن "الأدوية" أو "الدوا" أو "العلاج"، استخدم قسم الأدوية الحالية.
- عندما يسأل عن "الحساسية"، استخدم قسم الحساسية.
- عندما يسأل عن "الأمراض المزمنة" أو "التاريخ المرضي"، استخدم قسم الأمراض المزمنة.
""".strip()

    return f"""
Current patient context from Odoo:

Identity:
- Name: {_value_or_na(identity.get("name"), user_language)}
- Record ID: {_value_or_na(identity.get("id"), user_language)}
- Reference: {_value_or_na(identity.get("reference"), user_language)}
- SSN/ID: {_value_or_na(identity.get("ssn"), user_language)}
- Age: {_value_or_na(identity.get("age"), user_language)}
- Sex: {_value_or_na(identity.get("sex"), user_language)}
- Blood type: {_value_or_na(identity.get("blood_type"), user_language)}
- Date of birth: {_value_or_na(identity.get("dob"), user_language)}
- Nationality: {_value_or_na(identity.get("nationality"), user_language)}
- Marital status: {_value_or_na(identity.get("marital_status"), user_language)}
- File state: {_value_or_na(identity.get("state"), user_language)}
- Patient condition: {_value_or_na(identity.get("patient_condition"), user_language)}
- Admitted status: {_value_or_na(identity.get("patient_admitted"), user_language)}

Latest vital signs:
- Temperature: {_value_or_na(temp.get("value"), user_language)} °C, date: {_value_or_na(temp.get("date"), user_language)}
- Blood pressure: {_value_or_na(bp.get("systolic"), user_language)}/{_value_or_na(bp.get("diastolic"), user_language)} mmHg, date: {_value_or_na(bp.get("systolic_date") or bp.get("diastolic_date"), user_language)}
- Systolic blood pressure: {_value_or_na(bp.get("systolic"), user_language)} mmHg
- Diastolic blood pressure: {_value_or_na(bp.get("diastolic"), user_language)} mmHg
- Heart rate / pulse: {_value_or_na(hr.get("value"), user_language)} bpm, date: {_value_or_na(hr.get("date"), user_language)}
- Oxygen saturation: {_value_or_na(oxygen.get("value"), user_language)}%, date: {_value_or_na(oxygen.get("date"), user_language)}
- Respiratory rate: {_value_or_na(rr.get("value"), user_language)} /min, date: {_value_or_na(rr.get("date"), user_language)}
- Glycemia / blood sugar: {_value_or_na(glycemia.get("value"), user_language)}, date: {_value_or_na(glycemia.get("date"), user_language)}
- Weight: {_value_or_na(weight.get("value"), user_language)} kg, date: {_value_or_na(weight.get("date"), user_language)}
- Height: {_value_or_na(height.get("value"), user_language)} cm, date: {_value_or_na(height.get("date"), user_language)}

Clinical context:
- Critical info: {_value_or_na(clinical.get("critical_info"), user_language)}
- General info: {_value_or_na(clinical.get("general_info"), user_language)}
- Last diagnosis: {_value_or_na(last_diagnosis.get("value"), user_language)}, date: {_value_or_na(last_diagnosis.get("date"), user_language)}
- Last treatment plan: {_value_or_na(last_treatment_plan.get("value"), user_language)}, date: {_value_or_na(last_treatment_plan.get("date"), user_language)}
- Last visit summary: {_value_or_na(last_visit_summary.get("value"), user_language)}, date: {_value_or_na(last_visit_summary.get("date"), user_language)}

Allergies:
{allergies_text}

Chronic conditions:
{chronic_text}

Current medications (Active):
{medications_text}

Emergency contact:
- Name: {_value_or_na(emergency.get("name"), user_language)}
- Mobile: {_value_or_na(emergency.get("mobile"), user_language)}
- Relation: {_value_or_na(emergency.get("relation"), user_language)}

Interpretation notes:
- If the user asks about "pressure" or "BP", use blood pressure.
- If the user asks about "pulse", use heart rate.
- If the user asks about "oxygen", use oxygen saturation.
- If the user asks about "blood sugar", use glycemia.
- If the user asks about "medications", "meds", "drugs" or "treatment", use the current medications section.
- If the user asks about "allergy" or "allergies", use the allergies section.
- If the user asks about "chronic diseases" or "medical history", use the chronic conditions section.
""".strip()

def _build_patient_direct_prompt(
    question: str,
    session: Dict[str, Any],
    dashboard_payload: Dict[str, Any],
    user_language: str,
) -> str:
    language_rule = (
        "أجب باللغة العربية فقط وبأسلوب طبيعي وواضح."
        if user_language == "ar"
        else "Answer only in English with a natural, clear style."
    )

    patient_readable_context = _build_patient_readable_context(
        patient_context=session.get("current_patient") or {},
        user_language=user_language,
    )

    return f"""
You are CareSense AI, a smart medical assistant connected to Odoo.

You are answering a free-form chat question about the currently loaded patient.

Very important:
- Answer directly and naturally.
- If the user asks yes/no, start with نعم/لا in Arabic or Yes/No in English.
- Use ONLY the readable patient context below.
- Do NOT say data is missing if it appears in the readable patient context.
- Do NOT invent data outside the readable patient context.
- If a requested value is missing/null, say it is not available in the current Odoo record.
- You may cautiously interpret vital signs, but do not provide a confirmed diagnosis.
- If values look concerning, recommend clinical review or a doctor appointment.
- Do not expose raw JSON unless the user asks for raw data.
- {language_rule}

Readable patient context:
{patient_readable_context}

Recent chat history:
{_safe_json(session.get("history", [])[-5:])}

User question:
{question}
""".strip()


def _answer_from_loaded_patient_directly(
    session_id: str,
    session: Dict[str, Any],
    question: str,
    dashboard_payload: Dict[str, Any],
    user_language: str,
) -> Dict[str, Any]:
    model_name = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    llm = ChatOllama(
        model=model_name,
        base_url=base_url,
        temperature=0.2,
    )

    prompt = _build_patient_direct_prompt(
        question=question,
        session=session,
        dashboard_payload=dashboard_payload,
        user_language=user_language,
    )

    try:
        result = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=question),
        ])

        answer = result.content if hasattr(result, "content") else str(result)

        if not answer:
            answer = (
                "لم أستطع تكوين إجابة واضحة من بيانات المريض الحالية."
                if user_language == "ar"
                else "I could not produce a clear answer from the current patient data."
            )

        _append_history(session, question, answer)

        return _build_response(
            session_id=session_id,
            question=question,
            answer=answer,
            intent="patient_context_chat",
            provider="ollama",
            model=model_name,
            context={
                "current_patient_id": session.get("current_patient_id"),
                "current_patient": session.get("current_patient"),
                "history": session.get("history", [])[-5:],
            },
            trace={
                "framework": "direct_llm_patient_context",
                "reasoning_mode": "patient_context_to_llm_no_react_agent",
                "odoo_enabled": True,
                "error": None,
            },
        )

    except Exception as e:
        answer = (
            f"حدث خطأ أثناء تحليل بيانات المريض: {str(e)}"
            if user_language == "ar"
            else f"An error occurred while analyzing patient data: {str(e)}"
        )

        _append_history(session, question, answer)

        return _build_response(
            session_id=session_id,
            question=question,
            answer=answer,
            intent="patient_context_chat_error",
            provider="ollama",
            model=model_name,
            context={
                "current_patient_id": session.get("current_patient_id"),
                "current_patient": session.get("current_patient"),
                "history": session.get("history", [])[-5:],
            },
            trace={
                "framework": "direct_llm_patient_context",
                "reasoning_mode": "patient_context_to_llm_no_react_agent",
                "odoo_enabled": True,
                "error": str(e),
            },
        )


# ======================================================================
# Optional LangGraph / Tools path for non-patient dashboard questions
# ======================================================================

def _build_system_prompt(
    session: Dict[str, Any],
    dashboard_payload: Dict[str, Any],
    user_language: str,
) -> str:
    language_rule = (
        "أجب باللغة العربية فقط وبأسلوب طبيعي وواضح."
        if user_language == "ar"
        else "Answer only in English with a natural, clear style."
    )

    current_patient = session.get("current_patient")
    current_patient_brief = _patient_identity(current_patient)

    return f"""
You are CareSense AI, a smart medical AI agent connected to Odoo and a WiFi-sensing dashboard.

You are a free conversational assistant, not a keyword bot.

Rules:
- Chat naturally.
- Patient loading is handled by deterministic code before this step.
- If patient context is already provided, you may answer from it.
- Only use tools if you need dashboard or refreshed patient context.
- If the user asks yes/no, start with نعم/لا in Arabic or Yes/No in English.
- Do not invent patient data.
- {language_rule}

Current loaded patient brief:
{_safe_json(current_patient_brief)}

Recent chat history:
{_safe_json(session.get("history", [])[-5:])}

Dashboard payload:
{_safe_json(dashboard_payload)}
"""


def _make_tools(
    session: Dict[str, Any],
    dashboard_payload: Dict[str, Any],
):
    service = OdooPatientService()

    @tool
    def get_current_patient_context() -> str:
        """
        Get the currently loaded patient context from the chat session.
        """
        current_patient = session.get("current_patient")

        if not current_patient:
            return _safe_json({
                "status": "no_patient_loaded",
                "message": "No patient is currently loaded.",
            })

        return _safe_json({
            "status": "ok",
            "current_patient": current_patient,
        })

    @tool
    def get_dashboard_context() -> str:
        """
        Get the current simulated WiFi-sensing dashboard context.
        """
        return _safe_json({
            "status": "ok",
            "dashboard_payload": dashboard_payload,
        })

    @tool
    def refresh_current_patient_from_odoo() -> str:
        """
        Refresh the currently loaded patient context from Odoo using the current patient id.
        """
        patient_id = session.get("current_patient_id")

        if not patient_id:
            return _safe_json({
                "status": "no_patient_loaded",
                "message": "No patient is currently loaded.",
            })

        patient_context = _get_patient_context(service, patient_id)

        if not patient_context:
            return _safe_json({
                "status": "not_found",
                "patient_id": patient_id,
                "message": "Could not refresh patient from Odoo.",
            })

        session["current_patient"] = patient_context

        return _safe_json({
            "status": "ok",
            "message": "Current patient refreshed from Odoo.",
            "current_patient": patient_context,
        })

    return [
        get_current_patient_context,
        get_dashboard_context,
        refresh_current_patient_from_odoo,
    ]


# ======================================================================
# Formatting / Response
# ======================================================================

def _format_patient_selection(
    summaries: List[Dict[str, Any]],
    user_language: str,
) -> str:
    if user_language == "ar":
        lines = ["وجدت أكثر من مريض مطابق. من فضلك حدد المريض برقم السجل:"]
        for item in summaries:
            identity = _patient_identity(item)
            lines.append(
                f"- رقم السجل: {identity.get('id')}, "
                f"الاسم: {identity.get('name') or 'غير متوفر'}, "
                f"المرجع: {identity.get('reference') or 'غير متوفر'}, "
                f"رقم الهوية: {identity.get('ssn') or 'غير متوفر'}"
            )
        return "\n".join(lines)

    lines = ["I found more than one matching patient. Please specify the record ID:"]
    for item in summaries:
        identity = _patient_identity(item)
        lines.append(
            f"- Record ID: {identity.get('id')}, "
            f"Name: {identity.get('name') or 'N/A'}, "
            f"Reference: {identity.get('reference') or 'N/A'}, "
            f"ID Number: {identity.get('ssn') or 'N/A'}"
        )

    return "\n".join(lines)


def _build_response(
    session_id: str,
    question: str,
    answer: str,
    intent: str,
    provider: str,
    model: Optional[str],
    context: Dict[str, Any],
    trace: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "session_id": session_id,
        "agent": "CareSense Hybrid LangGraph Medical AI Agent",
        "provider": provider,
        "model": model,
        "question": question,
        "intent": intent,
        "answer": answer,
        "context": context,
        "agent_trace": {
            "timestamp": datetime.utcnow().isoformat(),
            **trace,
        },
        "disclaimer": (
            "Patient data is retrieved from Odoo according to configured permissions. "
            "This is not a confirmed medical diagnosis."
        ),
    }


# ======================================================================
# Public Entry Point
# ======================================================================

def generate_langgraph_chat_response(
    question: str,
    dashboard_payload: dict,
    session_id: str,
) -> Dict[str, Any]:
    question = (question or "").strip()
    user_language = _detect_language(question)
    session = _get_session(session_id)

    if not question:
        answer = (
            "من فضلك اكتب سؤالك."
            if user_language == "ar"
            else "Please write your question."
        )

        _append_history(session, question, answer)

        return _build_response(
            session_id=session_id,
            question=question,
            answer=answer,
            intent="empty",
            provider="system",
            model=None,
            context={
                "current_patient_id": session.get("current_patient_id"),
                "current_patient": session.get("current_patient"),
            },
            trace={
                "framework": "system",
                "reasoning_mode": "empty_message",
                "odoo_enabled": True,
                "error": None,
            },
        )

    patient_query = _extract_patient_load_query(question)

    if patient_query:
        return _load_patient_directly(
            session_id=session_id,
            session=session,
            question=question,
            patient_query=patient_query,
            user_language=user_language,
        )

    selected_patient_id = _extract_patient_id_selection(question)

    if selected_patient_id:
        return _select_patient_directly(
            session_id=session_id,
            session=session,
            question=question,
            patient_id=selected_patient_id,
            user_language=user_language,
        )

    if session.get("current_patient"):
        return _answer_from_loaded_patient_directly(
            session_id=session_id,
            session=session,
            question=question,
            dashboard_payload=dashboard_payload,
            user_language=user_language,
        )

    model_name = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    llm = ChatOllama(
        model=model_name,
        base_url=base_url,
        temperature=0.2,
    )

    tools = _make_tools(
        session=session,
        dashboard_payload=dashboard_payload,
    )

    agent = create_react_agent(
        model=llm,
        tools=tools,
    )

    system_prompt = _build_system_prompt(
        session=session,
        dashboard_payload=dashboard_payload,
        user_language=user_language,
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question),
    ]

    try:
        result = agent.invoke({"messages": messages})

        final_messages = result.get("messages", [])
        answer = ""

        for msg in reversed(final_messages):
            if isinstance(msg, AIMessage) and msg.content:
                answer = msg.content
                break

        if not answer:
            answer = (
                "لم أستطع تكوين إجابة واضحة."
                if user_language == "ar"
                else "I could not produce a clear answer."
            )

        _append_history(session, question, answer)

        return _build_response(
            session_id=session_id,
            question=question,
            answer=answer,
            intent="hybrid_langgraph_chat",
            provider="ollama",
            model=model_name,
            context={
                "current_patient_id": session.get("current_patient_id"),
                "current_patient": session.get("current_patient"),
                "history": session.get("history", [])[-5:],
            },
            trace={
                "framework": "langgraph",
                "reasoning_mode": "react_agent_for_non_patient_chat",
                "tools_enabled": [
                    "get_current_patient_context",
                    "get_dashboard_context",
                    "refresh_current_patient_from_odoo",
                ],
                "odoo_enabled": True,
                "error": None,
            },
        )

    except Exception as e:
        answer = (
            f"حدث خطأ أثناء تشغيل LangGraph Agent: {str(e)}"
            if user_language == "ar"
            else f"LangGraph Agent error: {str(e)}"
        )

        _append_history(session, question, answer)

        return _build_response(
            session_id=session_id,
            question=question,
            answer=answer,
            intent="hybrid_langgraph_error",
            provider="ollama",
            model=model_name,
            context={
                "current_patient_id": session.get("current_patient_id"),
                "current_patient": session.get("current_patient"),
                "history": session.get("history", [])[-5:],
            },
            trace={
                "framework": "langgraph",
                "reasoning_mode": "react_agent_for_non_patient_chat",
                "odoo_enabled": True,
                "error": str(e),
            },
        )
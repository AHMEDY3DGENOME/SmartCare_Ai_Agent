from typing import Any, Dict, List, Optional

from integrations.odoo.client import OdooClient


PATIENT_MODEL = "sm.patient"


PATIENT_BASIC_FIELDS = [
    "id",
    "name",
    "reference",
    "ssn",
    "mobile",
    "email",
    "age",
    "sex",
    "state",
    "patient_admitted",
    "patient_condition",
    "blood_type",
]


PATIENT_DETAIL_FIELDS = [
    "id",
    "name",
    "reference",
    "ssn",
    "mobile",
    "email",
    "age",
    "sex",
    "state",
    "patient_admitted",
    "patient_condition",
    "blood_type",
    "critical_info",
    "general_info",
    "emergency_name",
    "emergency_mobile",
    "emergency_contact_relation",
    "last_diagnosis",
    "diagnosis_date",
    "last_treatment_plan",
    "last_treatment_plan_date",
    "last_visit_summary",
    "last_visit_summary_date",
]


PATIENT_VITAL_FIELDS = [
    "id",
    "name",
    "last_temperature",
    "last_temperature_date",
    "last_systolic",
    "last_systolic_date",
    "last_diastolic",
    "last_diastolic_date",
    "last_glycemia",
    "last_glycemia_date",
    "last_heart_rate",
    "last_heart_rate_date",
    "last_oxygen_saturation",
    "last_oxygen_saturation_date",
    "last_respiratory_rate",
    "last_respiratory_rate_date",
    "last_weight",
    "last_weight_date",
    "last_height",
    "last_height_date",
]


def _clean_value(value: Any) -> Any:
    if value is False:
        return None

    if isinstance(value, list) and len(value) == 2:
        return {
            "id": value[0],
            "name": value[1],
        }

    return value


def _clean_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: _clean_value(value)
        for key, value in record.items()
    }


class OdooPatientService:
    def __init__(self):
        self.client = OdooClient()

    def search_patient(
        self,
        query: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:

        query = (query or "").strip()

        if not query:
            return []

        domain = [
            "|", "|", "|",
            ("name", "ilike", query),
            ("first_name", "ilike", query),
            ("last_name", "ilike", query),
            ("ssn", "ilike", query),
        ]

        records = self.client.search_read(
            model=PATIENT_MODEL,
            domain=domain,
            fields=PATIENT_BASIC_FIELDS,
            limit=limit,
            order="id desc",
        )

        return [_clean_record(record) for record in records]

    def get_patient_profile(
        self,
        patient_id: int,
    ) -> Optional[Dict[str, Any]]:

        records = self.client.read(
            model=PATIENT_MODEL,
            ids=[patient_id],
            fields=PATIENT_DETAIL_FIELDS,
        )

        if not records:
            return None

        return _clean_record(records[0])

    def get_patient_latest_vitals(
        self,
        patient_id: int,
    ) -> Optional[Dict[str, Any]]:

        records = self.client.read(
            model=PATIENT_MODEL,
            ids=[patient_id],
            fields=PATIENT_VITAL_FIELDS,
        )

        if not records:
            return None

        return _clean_record(records[0])

    def get_patient_full_summary(
        self,
        patient_id: int,
    ) -> Optional[Dict[str, Any]]:

        profile = self.get_patient_profile(patient_id)
        vitals = self.get_patient_latest_vitals(patient_id)

        if not profile:
            return None

        return {
            "profile": profile,
            "latest_vitals": vitals,
        }
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
    "last_height",
    "last_weight_date",
    "last_height_date",
]


ALL_PATIENT_FIELDS = list(
    dict.fromkeys(
        PATIENT_DETAIL_FIELDS + PATIENT_VITAL_FIELDS
    )
)


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
            "|", "|", "|", "|", "|",
            ("name", "ilike", query),
            ("first_name", "ilike", query),
            ("last_name", "ilike", query),
            ("ssn", "ilike", query),
            ("reference", "ilike", query),
            ("mobile", "ilike", query),
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

    def get_patient_record(
        self,
        patient_id: int,
    ) -> Optional[Dict[str, Any]]:
        records = self.client.read(
            model=PATIENT_MODEL,
            ids=[patient_id],
            fields=ALL_PATIENT_FIELDS,
        )

        if not records:
            return None

        return _clean_record(records[0])

    def get_patient_full_summary(
        self,
        patient_id: int,
    ) -> Optional[Dict[str, Any]]:
        record = self.get_patient_record(patient_id)

        if not record:
            return None

        return {
            "profile": self._extract_profile(record),
            "latest_vitals": self._extract_latest_vitals(record),
            "clinical_context": self._extract_clinical_context(record),
            "emergency_contact": self._extract_emergency_contact(record),
            "available_fields": list(record.keys()),
            "raw_record": record,
        }

    def get_patient_context_for_chat(
        self,
        patient_id: int,
    ) -> Optional[Dict[str, Any]]:
        """
        This is the main method used by the AI Agent.

        It returns a structured patient context that the LLM can use
        to answer any free-form Arabic or English question about the patient.
        """

        summary = self.get_patient_full_summary(patient_id)

        if not summary:
            return None

        return {
            "patient_identity": summary.get("profile"),
            "latest_vitals": summary.get("latest_vitals"),
            "clinical_context": summary.get("clinical_context"),
            "emergency_contact": summary.get("emergency_contact"),
            "source": {
                "system": "odoo",
                "model": PATIENT_MODEL,
                "patient_id": patient_id,
            },
            "limitations": [
                "Only fields configured in this service are available.",
                "If medication, lab results, allergies, or appointments are stored in separate Odoo models, add dedicated service methods for them.",
            ],
        }

    def get_patient_field_map(
        self,
        patient_id: int,
    ) -> Optional[Dict[str, Any]]:
        record = self.get_patient_record(patient_id)

        if not record:
            return None

        return {
            "patient_id": patient_id,
            "fields": record,
        }

    def _extract_profile(
        self,
        record: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "id": record.get("id"),
            "name": record.get("name"),
            "reference": record.get("reference"),
            "ssn": record.get("ssn"),
            "mobile": record.get("mobile"),
            "email": record.get("email"),
            "age": record.get("age"),
            "sex": record.get("sex"),
            "state": record.get("state"),
            "patient_admitted": record.get("patient_admitted"),
            "patient_condition": record.get("patient_condition"),
            "blood_type": record.get("blood_type"),
        }

    def _extract_latest_vitals(
        self,
        record: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "temperature": {
                "value": record.get("last_temperature"),
                "date": record.get("last_temperature_date"),
            },
            "blood_pressure": {
                "systolic": record.get("last_systolic"),
                "systolic_date": record.get("last_systolic_date"),
                "diastolic": record.get("last_diastolic"),
                "diastolic_date": record.get("last_diastolic_date"),
            },
            "glycemia": {
                "value": record.get("last_glycemia"),
                "date": record.get("last_glycemia_date"),
            },
            "heart_rate": {
                "value": record.get("last_heart_rate"),
                "date": record.get("last_heart_rate_date"),
            },
            "oxygen_saturation": {
                "value": record.get("last_oxygen_saturation"),
                "date": record.get("last_oxygen_saturation_date"),
            },
            "respiratory_rate": {
                "value": record.get("last_respiratory_rate"),
                "date": record.get("last_respiratory_rate_date"),
            },
            "weight": {
                "value": record.get("last_weight"),
                "date": record.get("last_weight_date"),
            },
            "height": {
                "value": record.get("last_height"),
                "date": record.get("last_height_date"),
            },
        }

    def _extract_clinical_context(
        self,
        record: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "critical_info": record.get("critical_info"),
            "general_info": record.get("general_info"),
            "last_diagnosis": {
                "value": record.get("last_diagnosis"),
                "date": record.get("diagnosis_date"),
            },
            "last_treatment_plan": {
                "value": record.get("last_treatment_plan"),
                "date": record.get("last_treatment_plan_date"),
            },
            "last_visit_summary": {
                "value": record.get("last_visit_summary"),
                "date": record.get("last_visit_summary_date"),
            },
        }

    def _extract_emergency_contact(
        self,
        record: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "name": record.get("emergency_name"),
            "mobile": record.get("emergency_mobile"),
            "relation": record.get("emergency_contact_relation"),
        }
from integrations.odoo.patients import OdooPatientService


service = OdooPatientService()

print("Search patient:")
patients = service.search_patient("Ahmed")

for patient in patients:
    print(patient)

if patients:
    patient_id = patients[0]["id"]

    print("\nPatient profile:")
    print(service.get_patient_profile(patient_id))

    print("\nLatest vitals:")
    print(service.get_patient_latest_vitals(patient_id))

    print("\nFull summary:")
    print(service.get_patient_full_summary(patient_id))

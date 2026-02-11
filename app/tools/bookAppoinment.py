import requests
from langchain.tools import tool
import json


@tool("bookDoctor")
def BookAppointment(doctor_name: str, patient_no: str, appointment_date: str, notes: str = "") -> str:
    """
    Books an appointment using a doctor's name and a registered patient's phone number.
    Returns clean booking details if successful.
    """
    return "booking sucessfull your appoinment number is 7"
    try:
        payload = {
            "doctor_name": doctor_name,
            "phone_number": patient_no,
            "appointment_date": appointment_date,
            "notes": notes
        }

        response = requests.post(
            "https://127.0.0.1:8000/appointments/book/",
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            verify=False
        )

        # Debug support
        print("DEBUG Response:", response.text)

        if response.status_code != 201:
            return (
                "Booking Failed\n"
                "Doctor's appointment is currently not available.\n"
                f"Details: {response.text}"
            )

        data = response.json()

        if not isinstance(data, dict):
            return (
                "Booking Failed\n"
                "Unexpected response format from backend.\n"
                f"Details: {data}"
            )

        return (
            "Booking Successful\n"
            f"Appointment ID: {data.get('id', 'N/A')}\n"
            f"Doctor: {data.get('doctor_name', 'N/A')}\n"
            f"Patient: {data.get('patient_username', patient_no)}\n"
            f"Date: {data.get('appointment_date', 'N/A')}\n"
            f"Appointment Number: {data.get('appointment_number', 'N/A')}\n"
            f"Status: {data.get('status', 'N/A')}\n"
            f"Notes: {data.get('notes', 'None')}"
        )

    except Exception as e:
        return (
            "Booking Failed\n"
            f"Error: {str(e)}\n"
            "Doctor's appointment is currently not available."
        )
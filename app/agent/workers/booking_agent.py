import re
import json
from typing import Optional, Dict, Any, List
from langchain.tools import tool
from app.agent.state import AgentState
from app.agent.logger import logger

# --- Tools ---

@tool
def fetch_hospitals(pincode: str, speciality: str = "general medicine"):
    """
    Fetch the list of hospitals based upon the pincode and required specialization of doctor.
    """
    # Mock data for demonstration
    return """
    1: Ganga Hospital, Singanallur
    2: ESI Hospital, Ukkadam
    3: Government Hospital, Railway Station Road
    """

@tool("bookAppointment")
def book_appointment_tool(hospital_id: str, patient_no: str, appointment_date: str) -> str:
    """
    Books an appointment using a hospital ID, phone number, and date.
    """
    return f"Booking Successful! Doctor Rajesh is allocated for you at the selected hospital. Your token number is 7. You will receive a call 30-45 mins prior to your appointment on {appointment_date}."

# --- Helpers ---

def extract_pincode(text: str) -> Optional[str]:
    match = re.search(r"\b\d{6}\b", text)
    return match.group(0) if match else None

def extract_phone(text: str) -> Optional[str]:
    # Match standard 10 digit phone numbers
    match = re.search(r"\b\d{10}\b", text)
    return match.group(0) if match else None

def extract_number(text: str) -> Optional[int]:
    match = re.search(r"\b(\d+)\b", text)
    return int(match.group(1)) if match else None

# --- Main Agent ---

async def booking_agent(state: AgentState) -> AgentState:
    """
    Stateful booking agent that guides the user through fetching hospitals 
    and booking an appointment.
    """
    # 1. Initialize Booking State if not exists
    if not state.get("booking_state"):
        state["booking_state"] = {
            "pincode": None,
            "phone_number": None,
            "preferred_type": None, # "nearby" | "low_cost"
            "hospitals": None,
            "selected_hospital_id": None,
            "confirmation": False,
            "booking_result": None,
            "appointment_date": "2024-02-15" # Default for now
        }
    
    bs = state["booking_state"]
    user_msg = (state.get("query") or "").lower().strip()
    urgency = state.get("urgency_score", 0)

    # STEP 1: PINCODE
    if not bs.get("pincode"):
        pincode = extract_pincode(user_msg)
        if pincode:
            bs["pincode"] = pincode
            logger.info(f"Pincode extracted: {pincode}")
        else:
            state["agent_response"] = "Please provide your 6-digit pincode so I can find hospitals near you."
            return state

    # STEP 2: PHONE NUMBER
    if not bs.get("phone_number"):
        phone = extract_phone(user_msg)
        if phone:
            bs["phone_number"] = phone
            logger.info(f"Phone extracted: {phone}")
        else:
            state["agent_response"] = "I'll need your 10-digit phone number to register the appointment."
            return state

    # STEP 3: PREFERRED TYPE (Nearby vs Low Cost)
    if not bs.get("preferred_type"):
        if urgency >= 70:
            bs["preferred_type"] = "nearby"
            logger.info("High urgency detected, defaulting to nearby hospitals.")
        else:
            if "near" in user_msg:
                bs["preferred_type"] = "nearby"
            elif "low" in user_msg or "cheap" in user_msg or "cost" in user_msg:
                bs["preferred_type"] = "low_cost"
            else:
                state["agent_response"] = "Would you prefer the nearest hospital or one that is low cost?"
                return state

    # STEP 4: FETCH & LIST HOSPITALS
    if not bs.get("hospitals"):
        raw_output = fetch_hospitals.invoke({
            "pincode": bs["pincode"],
        })

        hospitals = []
        # Parse the mock string response into a clean list
        lines = [l.strip() for l in raw_output.split("\n") if l.strip()]
        for line in lines:
            if ":" in line:
                h_id, name = line.split(":", 1)
                hospitals.append({
                    "id": h_id.strip(),
                    "name": name.strip()
                })
        
        if not hospitals:
            state["agent_response"] = "I couldn't find any hospitals in that area. Could you double-check the pincode?"
            bs["pincode"] = None # Reset pincode for retry
            return state

        bs["hospitals"] = hospitals
        options = "\n".join([f"{h['id']}. {h['name']}" for h in hospitals])
        state["agent_response"] = f"Here are the hospitals I found in {bs['pincode']}:\n\n{options}\n\nPlease reply with the number of the hospital you'd like to select."
        return state

    # STEP 5: HOSPITAL SELECTION
    if not bs.get("selected_hospital_id"):
        choice = extract_number(user_msg)
        valid_ids = [h["id"] for h in bs["hospitals"]]
        
        if not choice or str(choice) not in valid_ids:
            state["agent_response"] = "That wasn't a valid selection. Please choose a hospital number from the list above."
            return state

        selected = next(h for h in bs["hospitals"] if h["id"] == str(choice))
        bs["selected_hospital_id"] = selected["id"]
        bs["selected_hospital_name"] = selected["name"]

        state["agent_response"] = (
            f"You've selected {selected['name']}. Shall I go ahead and book the appointment for you? (Yes/No)"
        )
        return state

    # STEP 6: CONFIRMATION & FINAL ACTION
    if not bs.get("confirmation"):
        if "yes" in user_msg or "confirm" in user_msg or "ok" in user_msg:
            bs["confirmation"] = True
            
            # Execute the tool
            result = book_appointment_tool.invoke({
                "hospital_id": bs["selected_hospital_id"],
                "patient_no": bs["phone_number"],
                "appointment_date": bs["appointment_date"]
            })
            
            bs["booking_result"] = result
            state["agent_response"] = result
            state["final_response"] = result
            logger.info("Booking completed successfully.")
        elif "no" in user_msg:
            state["agent_response"] = "No problem. I've cancelled the booking process. Let me know if you need anything else."
            state["booking_state"] = None # Reset state
        else:
            state["agent_response"] = f"I'm ready to book your appointment at {bs.get('selected_hospital_name')}. Please confirm with a 'Yes' or 'No'."
            
    return state
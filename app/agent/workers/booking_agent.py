import json
import os
import re
import sys
import asyncio
from datetime import datetime
from app.agent.state import AgentState
from app.agent.llms.groq import get_groq_llm

llm=get_groq_llm()

# --- Prompts ---
SYSTEM_PROMPT = """
You are an intent extraction engine for a medical booking system.
Your ONLY task is to extract structured fields from the conversation.
You are NOT a chat assistant. You must NOT ask questions or explain anything.
You must output STRICT JSON ONLY.

Output MUST be a single JSON object with EXACTLY these keys:
{
  "pincode": string or null (6 digits),
  "hospital_id": number or null,
  "appointment_date": string or null (YYYY-MM-DD),
  "booking_confirmed": boolean or null
}

Rules:
- Pincodes MUST be 6-digit strings.
- Convert ALL number words to digits (e.g., "six one two" -> "612", "sixty" -> "60", "oh" -> "0").
- If user says "six one two zero zero four", output "612004".
- If multiple numbers present, prefer the one explicitly reduced to 6 digits.
- If there are two pincodes, extract the MORE RECENT one.
- Convert relative dates (tomorrow, etc.) to YYYY-MM-DD using current_date.
- hospital_id must be chosen from hospital_options if provided.
- Map "first", "1", "one" to corresponding hospital_id.
- booking_confirmed is true ONLY if user explicitly says "yes/confirm".
"""

# --- Nodes ---
def emit(text: str) -> dict:
    return {
        "messages": [{"role": "assistant", "content": text}],
        "final_response": text,
        "agent_response": text,   # 👈 REQUIRED
    }
async def extract_intent(state: AgentState) -> dict:
    print("reached extract intent ")
    messages = state.get("messages", [])[-10:]
    print("messages : ", messages)
    user_context = {
        "role": "user",
        "content": json.dumps({
            "conversation": messages,
            "current_date": datetime.now().strftime("%Y-%m-%d"),
            "known_state": {
                "pincode": state.get("pincode"),
                "hospital_id": state.get("hospital_id"),
                "appointment_date": state.get("appointment_date"),
                "hospital_options": state.get("hospital_options"),
            }
        })
    }

    try:
        response = await llm.ainvoke(
            [{"role": "system", "content": SYSTEM_PROMPT}, user_context]
        )
        content = response.content
        print("LLM RESPONSE",content)
        
        # Clean up markdown code blocks if present
        if "```" in content:
            content = re.sub(r"```json\s*", "", content)
            content = re.sub(r"```", "", content)

        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            content = match.group(0)

        intent = json.loads(content)

        updates = {}  # ⚠️ DO NOT touch final_response here
        updates["current_flow"] = "booking" # PERSIST FLOW

        if intent.get("pincode"):
            updates["pincode"] = str(intent["pincode"])

        if intent.get("hospital_id"):
            hid = intent["hospital_id"]
            updates["hospital_id"] = int(hid[0]) if isinstance(hid, list) else int(hid)

        if intent.get("appointment_date"):
            updates["appointment_date"] = intent["appointment_date"]

        if "booking_confirmed" in intent and intent["booking_confirmed"] is not None:
            updates["booking_confirmed"] = intent["booking_confirmed"]
        print("updates : ", updates)
        return updates

    except Exception as e:
        print(f"DEBUG: [extract_intent] Error: {e}")
        return emit("I'm having trouble understanding. Could you please repeat?")
    

def decide_next_step(state: AgentState) -> str:
    pincode = state.get("pincode")
    hospitals = state.get("hospital_options")
    hospital_id = state.get("hospital_id")
    date = state.get("appointment_date")
    confirmed = state.get("booking_confirmed")
    
    # If we are here, we are in booking flow
    # This might have been set by medical agent, but ensure it stays set
    # Note: We can't easily mutate state here in a conditional edge function in all LangGraph versions,
    # but the node following this will run in the context where we can't miss it.
    # Actually, specific nodes should probably set it. 

    if not pincode: return "ask_pincode"
    if not hospitals: return "fetch_hospitals"
    if not hospital_id: return "ask_hospital_selection"
    if not date: return "ask_appointment_date"
    if confirmed is None: return "ask_confirmation"

    return "perform_booking"

async def fetch_hospitals(state: AgentState) -> dict:
    print("reached fetch_hospitals :")
    try:
        res = await state["mcp_client"].call_tool(
            "fetch_hospitals",
            {"pincode": state["pincode"]}
        )

        raw_text = res[0].text
        data = json.loads(raw_text)

        hospitals = data.get("hospitals")

        return {
            "hospital_options": hospitals
        }

    except json.JSONDecodeError:
        return emit("Received invalid data while fetching hospitals.")
    except Exception as e:
        return emit(f"Error fetching hospitals: {e}")
    

async def perform_booking(state: AgentState) -> dict:
    try:
        res = await state["mcp_client"].call_tool(
            "do_booking",
            {
                "hospital_id": state["hospital_id"],
                "patient_contact": state["patient_contact"],
                "appointment_date": state["appointment_date"],
                "urgency_score": state.get("urgency_score", 50),
            }
        )

        raw_text = res[0].text
        data = json.loads(raw_text)
        docter=data.get("doctor_details")
        if "appointment_id" in data:
            return {
                **emit(
                    "Appointment booked successfully!\n\n"
                    f"Doctor: {docter.get('doctor_name')}\n"
                    f"Date: {data.get('appointment_date')}\n"
                    f"Token No: {data.get('token_no')}\n"
                    f"Appointment ID: {data.get('appointment_id')}"
                ),
                "booking_confirmed": True,
            }

        return {
            **emit(f"❌ Booking failed: {data.get('message', 'Unknown error')}"),
            "booking_confirmed": None,
        }

    except json.JSONDecodeError:
        return emit("❌ Booking error: Invalid response format")
    except Exception as e:
        return emit(f"❌ Booking error: {e}")

def ask_pincode(state: AgentState) -> dict:
    return emit("Please provde  your 6-digit pincode.")


def ask_hospital_selection(state: AgentState) -> dict:
    hospitals = state.get("hospital_options", [])

    if not hospitals:
        return {
            "pincode": None,
            "hospital_options": None,
            "hospital_id": None,
            **emit("No hospitals found for that pincode. Please try another one."),
        }

    options = "\n".join([f"{h['id']}: {h['name']}" for h in hospitals])
    return emit(f"Please select a hospital ID {options}")

def ask_appointment_date(state: AgentState) -> dict:
    return emit("What date would you like to book for?")

def ask_confirmation(state: AgentState) -> dict:
    return emit(
        f"to Confirm booking for Hospital {state['hospital_id']} "
        f"on {state['appointment_date']} say yes "
    )
import asyncio
import websockets
import json

async def test_multi_agent():
    uri = "ws://localhost:8000/ws"
    
    async with websockets.connect(uri) as ws:
        # Test 1: Greeting (direct response)
        await ws.send(json.dumps({
            "type": "user_message",
            "content": "who are you "
        }))
        response = await ws.recv()
        print("Response 1:", json.loads(response)["content"])
        
        # Test 2: Symptom analysis
        await ws.send(json.dumps({
            "type": "user_message",
            "content": "I have a severe headache and fever"
        }))
        response = await ws.recv()
        print("Response 2:", json.loads(response)["content"])
        
        # Test 3: Booking
        await ws.send(json.dumps({
            "type": "user_message",
            "content": "I need to book an appointment for my symptoms"
        }))
        response = await ws.recv()
        print("Response 3:", json.loads(response)["content"])

asyncio.run(test_multi_agent())



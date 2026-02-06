import asyncio
import websockets
import json

async def test_ws():
    uri = "ws://localhost:8000/ws"
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected!")
            msg = {"type": "user_message", "content": "test message from script"}
            await websocket.send(json.dumps(msg))
            print("Sent message")
            while True:
                response = await websocket.recv()
                print(f"Received: {response}")
                if '"event": "done"' in response:
                    break
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_ws())

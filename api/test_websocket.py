import asyncio
import websockets
import json

async def test_evolution():
    uri = "ws://localhost:8000/ws/evolve/test-job-123"
    async with websockets.connect(uri) as websocket:
        # Send start command
        await websocket.send(json.dumps({"action": "start"}))
        
        print("Connected to Evolution Engine...")
        
        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)
                
                if data.get("status") == "Completed":
                    print("\nEvolution Completed Successfully!")
                    break
                    
                print(f"Gen {data.get('generation')}: Score {data.get('best_score')} | Status: {data.get('status')}")
                
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed")
                break

if __name__ == "__main__":
    asyncio.run(test_evolution())

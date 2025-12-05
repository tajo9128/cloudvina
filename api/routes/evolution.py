from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from ..services.evolution_engine import GeneticAlgorithm
import json
import asyncio

router = APIRouter()

@router.websocket("/ws/evolve/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for Real-Time Evolution.
    Connects the frontend to the Genetic Algorithm loop.
    """
    await websocket.accept()
    
    try:
        # 1. Receive initial configuration from client (e.g., target protein, seed molecule)
        data = await websocket.receive_text()
        config = json.loads(data)
        
        # Mock data for now - in production, fetch PDBQT from S3 using job_id
        receptor_content = "MOCK_PDBQT_CONTENT" 
        center = (0, 0, 0)
        size = (20, 20, 20)
        
        # 2. Initialize the Engine
        engine = GeneticAlgorithm(receptor_content, center, size)
        
        # 3. Initialize Population (using a simple seed for demo)
        # In production, use the ligand uploaded by the user
        seed_smiles = ["c1ccccc1"] # Benzene
        
        # 4. Run the Evolution Loop
        for result in engine.evolve(seed_smiles):
            # Send update to frontend
            result["status"] = "Running"
            await websocket.send_json(result)
            
            # Simulate processing time (remove in production)
            await asyncio.sleep(0.5) 
            
        # 5. Finish
        await websocket.send_json({"status": "Completed", "message": "Evolution finished."})
        await websocket.close()
        
    except WebSocketDisconnect:
        print(f"Client disconnected from job {job_id}")
    except Exception as e:
        print(f"Error in evolution loop: {str(e)}")
        await websocket.close(code=1011)

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from services.evolution_engine import GeneticAlgorithm
import json
import asyncio
import boto3
import os
from rdkit import Chem

router = APIRouter()

def parse_vina_config(config_content):
    """Parse Vina config file to get center and size."""
    config = {}
    for line in config_content.splitlines():
        if '=' in line:
            key, value = line.split('=')
            config[key.strip()] = float(value.strip())
    
    center = (config.get('center_x', 0), config.get('center_y', 0), config.get('center_z', 0))
    size = (config.get('size_x', 20), config.get('size_y', 20), config.get('size_z', 20))
    return center, size

@router.websocket("/ws/evolve/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for Real-Time Evolution.
    Connects the frontend to the Genetic Algorithm loop.
    """
    await websocket.accept()
    
    try:
        # 1. Receive initial handshake
        data = await websocket.receive_text()
        _ = json.loads(data)
        
        # 2. Fetch Job Data from S3
        s3 = boto3.client('s3', region_name=os.getenv("AWS_REGION", "us-east-1"))
        bucket = os.getenv("S3_BUCKET", "BioDockify-jobs-use1-1763775915")
        
        try:
            # Fetch Receptor
            # Try .pdb first, then .pdbqt
            try:
                receptor_obj = s3.get_object(Bucket=bucket, Key=f"{job_id}/receptor.pdb")
                receptor_content = receptor_obj['Body'].read().decode('utf-8')
            except:
                receptor_obj = s3.get_object(Bucket=bucket, Key=f"{job_id}/receptor.pdbqt")
                receptor_content = receptor_obj['Body'].read().decode('utf-8')

            # Fetch Config
            config_obj = s3.get_object(Bucket=bucket, Key=f"{job_id}/config.txt")
            config_content = config_obj['Body'].read().decode('utf-8')
            center, size = parse_vina_config(config_content)

            # Fetch Docked Ligand (Output)
            output_obj = s3.get_object(Bucket=bucket, Key=f"{job_id}/output.pdbqt")
            output_content = output_obj['Body'].read().decode('utf-8')
            
            # Extract SMILES from Docked Ligand
            # Use the first model in the PDBQT
            mol = Chem.MolFromPDBBlock(output_content)
            if mol:
                seed_smiles = [Chem.MolToSmiles(mol)]
            else:
                # Fallback if RDKit fails to parse PDBQT directly (common with PDBQT)
                # Try to fetch original ligand if possible, or error out
                # For now, let's try a fallback or just notify
                await websocket.send_json({"status": "Error", "message": "Could not parse docked ligand."})
                return

        except Exception as e:
            print(f"S3 Fetch Error: {e}")
            await websocket.send_json({"status": "Error", "message": f"Failed to load job data: {str(e)}"})
            await websocket.close()
            return

        # 3. Initialize the Engine
        # Auto-detect format (PDB or PDBQT)
        engine = GeneticAlgorithm(receptor_content, center, size, receptor_format='auto')
        
        # 4. Run the Evolution Loop
        for result in engine.evolve(seed_smiles):
            # Send update to frontend
            result["status"] = "Running"
            await websocket.send_json(result)
            
            # Small delay to not overwhelm frontend
            await asyncio.sleep(0.1) 
            
        # 5. Finish
        await websocket.send_json({"status": "Completed", "message": "Evolution finished."})
        await websocket.close()
        
    except WebSocketDisconnect:
        print(f"Client disconnected from job {job_id}")
    except Exception as e:
        print(f"Error in evolution loop: {str(e)}")
        try:
            await websocket.close(code=1011)
        except:
            pass

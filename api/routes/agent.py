from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any
# from api.agent_zero import AgentZeroClient  # Deprecated HF
from agent_zero.gemini_client import GeminiClient
from agent_zero import PROMPT_EXPLAIN_RESULTS, PROMPT_NEXT_STEPS

router = APIRouter(prefix="/agent", tags=["agent_zero"])
agent = GeminiClient()

class ConsultRequest(BaseModel):
    query: str
    context_type: str  # 'result_explanation', 'next_steps', 'general'
    data: Dict[str, Any]

@router.post("/consult")
async def consult_agent(request: ConsultRequest):
    """
    Consult Agent Zero for reasoning and guidance.
    """
    try:
        # Select prompt template based on context type
        if request.context_type == "result_explanation":
             user_prompt = PROMPT_EXPLAIN_RESULTS.format(
                 context_json=request.data
             )
        elif request.context_type == "next_steps":
             user_prompt = PROMPT_NEXT_STEPS.format(
                 context_json=request.data
             )
        elif request.context_type == "peer_review":
             # Import locally to avoid circulars if any, though PROMPT_PEER_REVIEW is in prompts
             from agent_zero.prompts import PROMPT_PEER_REVIEW
             user_prompt = PROMPT_PEER_REVIEW.format(
                 context_json=request.data
             )
        else:
             user_prompt = request.query

        # Call Llama 3.3
        response = agent.consult(
            prompt=user_prompt, 
            context=request.data
        )
        
        if "error" in response:
            raise HTTPException(status_code=503, detail=response["error"])

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tools/{tool_name}")
async def execute_tool(tool_name: str, args: Dict[str, Any]):
    """
    Directly execute an Agent Zero tool (UniProt, RCSB, etc.)
    returns structured JSON data.
    """
    try:
        # Re-use the execution logic from the client, or import directly
        # Since logic is inside the client class as private, let's expose specific imports
        from api.agent_zero.tools import fetch_target_profile, fetch_structure_metadata, fetch_bioactivity, fetch_pockets_for_pdb
        
        if tool_name == "uniprot":
            return fetch_target_profile(args.get("id"))
        elif tool_name == "rcsb":
            return fetch_structure_metadata(args.get("id"))
        elif tool_name == "chembl":
            return fetch_bioactivity(args.get("id"))
        elif tool_name == "pockets":
            return fetch_pockets_for_pdb(args.get("id"))
        elif tool_name == "prioritization":
            from api.agent_zero.tools import prioritize_leads
            # Requires body payload "data" with {leads: [], budget: float, risk: str}
            # For simplicity in this GET/POST structure, we expect args to contain them
            # But wait, execute_tool is POST. The body is args.
            
            leads = args.get("leads", [])
            budget = args.get("budget", 10000)
            risk = args.get("risk", "medium")
            return prioritize_leads(leads, budget, risk)
        
        elif tool_name == "developability":
             from api.agent_zero.tools import assess_developability
             # Expects "smiles" in args
             smiles = args.get("smiles")
             if not smiles:
                 return {"error": "Missing 'smiles' argument"}
             return assess_developability(smiles)

        else:
             raise HTTPException(400, "Unknown tool")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

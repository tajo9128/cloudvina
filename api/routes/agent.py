from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any
from api.agent_zero import AgentZeroClient, PROMPT_EXPLAIN_RESULTS, PROMPT_NEXT_STEPS

router = APIRouter(prefix="/agent", tags=["agent_zero"])
agent = AgentZeroClient()

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

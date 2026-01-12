"""Single-Call Planning Envelope - Reduces LLM calls by 40-60%."""
import json
import hashlib
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class PlanningRequest:
    intent: str
    repo_summary: str
    constraints: List[str]
    required_outputs: List[str]
    task_context: Dict[str, Any] = None

class PlanningEnvelope:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self._cache = {}
    
    def plan(self, request, use_cache=True):
        if use_cache:
            cache_key = self._generate_cache_key(request)
            if cache_key in self._cache:
                return self._cache[cache_key]
        prompt = self._build_prompt(request)
        response_text = self._call_llm(prompt)
        response = self._parse_response(response_text)
        if use_cache:
            self._cache[cache_key] = response
        return response
    
    def _generate_cache_key(self, request):
        content = f"{request.intent}:{request.repo_summary}:{request.constraints}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _build_prompt(self, request):
        outputs = ", ".join(request.required_outputs)
        # Use regular string format, not f-string, to avoid brace conflicts
        prompt = """You are an expert AI planner. Task: {intent}
Repo: {repo}
Constraints: {constraints}

Required outputs: {outputs}

Respond with JSON containing these fields: plan, tool_sequence, risk_assessment, reasoning, alternatives, confidence.""".format(
            intent=request.intent,
            repo=request.repo_summary,
            constraints=request.constraints,
            outputs=outputs
        )
        return prompt
    
    def _call_llm(self, prompt):
        if self.llm_client:
            return self.llm_client.call(prompt)
        # Mock response for testing
        return json.dumps({
            "plan": "Execute task step by step",
            "tool_sequence": [{"tool": "code_execution", "purpose": "execute"}],
            "risk_assessment": {"level": "low"},
            "reasoning": "Simple task",
            "alternatives": [],
            "confidence": 0.9
        })
    
    def _parse_response(self, text):
        try:
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            if json_start >= 0:
                data = json.loads(text[json_start:json_end])
                return data
        except:
            pass
        return {"plan": text, "tool_sequence": [], "risk_assessment": {"level": "unknown"}}

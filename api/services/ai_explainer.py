import os
import httpx
from typing import Dict, AsyncGenerator

class AIExplainer:
    """AI service for explaining docking results"""
    
    def __init__(self):
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        self.base_url = "https://api.deepseek.com"
        self.model = "deepseek-chat"  # Points to DeepSeek-V3
    
    async def explain_results(
        self, 
        job_data: Dict,
        analysis: Dict,
        interactions: Dict,
        user_question: str = None
    ) -> AsyncGenerator[str, None]:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": self._get_system_prompt()
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "stream": True,
                        "temperature": 0.7,
                        "max_tokens": 1000
                    },
                    timeout=30.0
                )
                
                
                if response.status_code == 429:
                    import json
                    yield f"data: {json.dumps('⏳ Rate limit reached. The AI service is temporarily busy. Please wait 30-60 seconds and try again, or use fewer requests per minute.')}\\n\\n"
                    return
                elif response.status_code != 200:
                    import json
                    error_detail = ""
                    try:
                        error_data = response.json()
                        error_detail = error_data.get('error', {}).get('message', '')
                    except:
                        pass
                    yield f"data: {json.dumps(f'Error: API returned status {response.status_code}. {error_detail}')}\\n\\n"
                    return

                # Stream response
                import json
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data_json = json.loads(data_str)
                            content = data_json.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if content:
                                yield f"data: {json.dumps(content)}\n\n"
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            yield f"data: {json.dumps('Error: ' + str(e))}\n\n"
    
    def _get_system_prompt(self) -> str:
        """System prompt for educational context"""
        return """You are a friendly molecular docking tutor for pharmacy and chemistry students.
Your job is to explain docking results in simple, clear language.

Guidelines:
1. Avoid jargon - when you must use technical terms, explain them
2. Use analogies (lock-and-key, puzzle pieces, etc.)
3. Focus on practical drug discovery meaning
4. Be encouraging and educational
5. Keep responses concise (2-3 paragraphs max)
6. Use emojis sparingly but appropriately

Remember: These are students learning drug design, not experts."""
    
    def _create_prompt(
        self,
        job_data: Dict,
        analysis: Dict,
        interactions: Dict,
        user_question: str = None
    ) -> str:
        """Create detailed prompt with results"""
        
        # Extract key data
        best_affinity = analysis.get('best_affinity', 'N/A') if analysis else 'N/A'
        num_poses = analysis.get('num_poses', 0) if analysis else 0
        receptor = job_data.get('receptor_filename', 'Unknown')
        ligand = job_data.get('ligand_filename', 'Unknown')
        
        # Interaction counts
        h_bonds = len(interactions.get('hydrogen_bonds', [])) if interactions else 0
        hydrophobic = len(interactions.get('hydrophobic_contacts', [])) if interactions else 0
        
        if user_question:
            # Answering specific question
            prompt = f"""Student Question: {user_question}

Docking Results Context:
- Protein: {receptor}
- Ligand: {ligand}
- Best Binding Affinity: {best_affinity} kcal/mol
- Docking Poses Found: {num_poses}
- Hydrogen Bonds: {h_bonds}
- Hydrophobic Contacts: {hydrophobic}

Please answer their question using this data."""
        else:
            # General explanation
            prompt = f"""Explain this molecular docking result to a pharmacy student:

**Experiment:**
- Protein Target: {receptor}
- Drug Candidate: {ligand}

**Results:**
- Best Binding Affinity: {best_affinity} kcal/mol
- Number of Docking Poses: {num_poses}
- Hydrogen Bonds Formed: {h_bonds}
- Hydrophobic Contacts: {hydrophobic}

**Please explain:**
1. What does this binding affinity mean? Is it good or bad?
2. What do the interactions tell us about how well the drug binds?
3. Would this be a promising drug candidate based on these results?

Keep it simple and educational!"""
        
        return prompt

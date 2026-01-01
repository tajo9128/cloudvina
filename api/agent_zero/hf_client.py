import os
import requests
import json
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Model Configuration
PRIMARY_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
BACKUP_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"

class AgentZeroClient:
    """
    Client for 'Agent Zero' - BioDockify's AI Reasoning Engine.
    Uses Hugging Face Inference API to access Llama 3.3 70B.
    """
    
    def __init__(self):
        self.api_key = os.getenv("HUGGINGFACE_API_KEY")
        self.api_url = "https://api-inference.huggingface.co/models"
        
        if not self.api_key:
            logger.warning("Agent Zero: HUGGINGFACE_API_KEY not found. AI features will be disabled.")

    def consult(self, prompt: str, context: dict = None, use_backup: bool = False) -> dict:
        """
        Send a reasoning request to the AI model.
        
        Args:
            prompt (str): The user's query or system prompt.
            context (dict): Optional context data (docking results, protein info).
            use_backup (bool): If True, use Mixtral instead of Llama 3.3.
            
        Returns:
            dict: The AI's JSON response containing 'analysis', 'suggestion', etc.
        """
        if not self.api_key:
            return {
                "error": "AI Agent not configured. Please add HUGGINGFACE_API_KEY to .env"
            }

        model = BACKUP_MODEL if use_backup else PRIMARY_MODEL
        url = f"{self.api_url}/{model}"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Construct a rich prompt that encourages reasoning
        full_prompt = self._construct_system_prompt(prompt, context)
        
        payload = {
            "inputs": full_prompt,
            "parameters": {
                "max_new_tokens": 1024,
                "temperature": 0.7, # Balanced creativity/logic
                "top_p": 0.9,
                "return_full_text": False
            }
        }

        try:
            logger.info(f"Agent Zero thinking... (Model: {model})")
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code != 200:
                # If primary model fails (e.g., loading), try backup once
                if not use_backup and response.status_code in [503, 504]:
                    logger.warning(f"Llama 3.3 busy (Status {response.status_code}). Switching to Mixtral backup.")
                    return self.consult(prompt, context, use_backup=True)
                
                logger.error(f"HF API Error {response.status_code}: {response.text}")
                return {"error": f"AI Service Unavailable ({response.status_code})"}

            result = response.json()
            
            # Parse the text response
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get('generated_text', '')
                return self._parse_response(generated_text)
            else:
                return {"error": "Invalid response format from AI"}

        except Exception as e:
            logger.error(f"Agent Zero exception: {str(e)}")
            return {"error": "Internal Agent Error"}

    def _construct_system_prompt(self, user_prompt: str, context: dict) -> str:
        """
        Builds the Llama 3 prompt format <|begin_of_text|><|start_header_id|>...
        """
        # Context string construction
        context_str = ""
        if context:
            context_str = f"CONTEXT DATA:\n{json.dumps(context, indent=2)}\n"

        # Llama 3 Instruct Format
        prompt_template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are Agent Zero, the expert AI research assistant for BioDockify.
Your goal is to provide scientific reasoning, interpret docking results, and suggest valid next steps.
You are NOT just a text generator; you are a reasoning engine.

RULES:
1. Be concise and scientific.
2. Don't make up data.
3. If data is missing, say so.
4. Output specific JSON format when requested.
<|eot_id|><|start_header_id|>user<|end_header_id|>

{context_str}
QUESTION: {user_prompt}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        
        return prompt_template

    def _parse_response(self, text: str) -> dict:
        """
        Attempts to extract JSON from the response text.
        If strict JSON fails, returns text mapped to a generic structure.
        """
        text = text.strip()
        
        # Try finding JSON block ```json ... ```
        if "```json" in text:
            try:
                start = text.index("```json") + 7
                end = text.rindex("```")
                json_str = text[start:end].strip()
                return json.loads(json_str)
            except:
                pass
        
        # Fallback: Just return text
        return {
            "analysis": text,
            "raw_text": True
        }

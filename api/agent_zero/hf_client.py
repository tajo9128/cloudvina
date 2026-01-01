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

    def consult(self, prompt: str, context: dict = None, use_backup: bool = False, depth: int = 0) -> dict:
        """
        Send a reasoning request to the AI model. 
        Auto-executes tools if requested (max recursion depth = 2).
        """
        if not self.api_key:
            return {"error": "AI Agent not configured. Please add HUGGINGFACE_API_KEY to .env"}
        
        # Prevent infinite loops
        if depth > 2:
            return {"error": "Agent Loop Limit Reached"}

        model = BACKUP_MODEL if use_backup else PRIMARY_MODEL
        url = f"{self.api_url}/{model}"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Dynamic System Prompt
        from .prompts import SYSTEM_PROMPT_CORE
        
        # Build prompt (inject previous context if needed, but for now simple)
        # Note: Ideally we pass full history, here we reconstruct for single turn or recursive turn
        # For simplicity in this codebase, we use the method logic to chain.
        
        # If this is a follow-up (depth > 0), the prompt argument usually contains the tool output info
        full_text = self._construct_prompt_text(SYSTEM_PROMPT_CORE, prompt, context)

        payload = {
            "inputs": full_text,
            "parameters": {
                "max_new_tokens": 1024,
                "temperature": 0.1 if depth > 0 else 0.7, # Lower temp for tool handling
                "top_p": 0.9,
                "return_full_text": False
            }
        }

        try:
            logger.info(f"Agent Zero thinking... (Model: {model} | Depth: {depth})")
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code != 200:
                if not use_backup and response.status_code in [503, 504]:
                    return self.consult(prompt, context, use_backup=True, depth=depth)
                return {"error": f"AI Service Unavailable ({response.status_code})"}

            result = response.json()
            generated_text = result[0].get('generated_text', '') if isinstance(result, list) else ''
            
            # PARSE RESPONSE
            parsed = self._parse_response(generated_text)
            
            # CHECK FOR TOOL CALL
            if parsed.get("tool"):
                tool_name = parsed["tool"]
                tool_args = parsed.get("args", {})
                logger.info(f"Agent requested tool: {tool_name} with {tool_args}")
                
                # Execute Tool
                tool_output = self._execute_tool(tool_name, tool_args)
                
                # Recursive Call with Tool Output
                # We append the tool result to the prompt and ask for interpretation.
                # Llama 3 Instruct format handles chat history, but for this simple wrapper we append.
                # New Prompt: "User: <Original> \n Assistant: <ToolCall> \n System: Tool Output: <Output>. Please interpret this."
                
                follow_up_prompt = f"{prompt}\n[AGENT CALLED TOOL: {tool_name}]\n[TOOL OUTPUT]: {json.dumps(tool_output)}\nPlease interpret this data for the user."
                
                # Recursive call
                return self.consult(follow_up_prompt, context, use_backup, depth=depth+1)
            
            return parsed

        except Exception as e:
            logger.error(f"Agent Zero exception: {str(e)}")
            return {"error": str(e)}

    def _execute_tool(self, tool_name: str, args: dict) -> dict:
        """Executes the requested tool by name"""
        try:
            from .tools import fetch_target_profile, fetch_structure_metadata, fetch_bioactivity, fetch_pockets_for_pdb
            
            if tool_name == "fetch_target_profile":
                return fetch_target_profile(args.get("uniprot_id"))
            elif tool_name == "fetch_structure_metadata":
                return fetch_structure_metadata(args.get("pdb_id"))
            elif tool_name == "fetch_bioactivity":
                return fetch_bioactivity(args.get("chembl_id"))
            elif tool_name == "fetch_pockets_for_pdb":
                return fetch_pockets_for_pdb(args.get("pdb_id"))
            
            return {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            return {"error": f"Tool Execution Failed: {str(e)}"}

    def _construct_prompt_text(self, system: str, user: str, context: dict) -> str:
        context_str = ""
        if context:
            context_str = f"CONTEXT DATA:\n{json.dumps(context, indent=2)}\n"

        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system}
<|eot_id|><|start_header_id|>user<|end_header_id|>

{context_str}
QUESTION: {user}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    def _parse_response(self, text: str) -> dict:
        text = text.strip()
        if "```json" in text:
            try:
                start = text.index("```json") + 7
                end = text.rindex("```")
                json_str = text[start:end].strip()
                return json.loads(json_str)
            except: pass
        
        # Also try to parse if the entire text is just JSON without backticks
        if text.startswith("{") and text.endswith("}"):
            try: return json.loads(text)
            except: pass
            
        return {"analysis": text, "raw_text": True}

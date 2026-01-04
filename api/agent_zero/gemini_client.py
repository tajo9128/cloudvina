
import os
import logging
import json
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

logger = logging.getLogger(__name__)

class GeminiClient:
    """
    Client for Agent Zero using Google's Gemini 1.5 Flash.
    Stable, Fast, and Free Tier available.
    """
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("Agent Zero: GEMINI_API_KEY not found. Agent disabled.")
        else:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')

    def consult(self, prompt: str, context: dict = None, use_backup: bool = False, depth: int = 0) -> dict:
        if not self.api_key:
            return {"error": "GEMINI_API_KEY not configured."}
        
        if depth > 2: return {"error": "Loop Limit"}

        # Construct Prompt
        from .prompts import SYSTEM_PROMPT_CORE
        full_text = self._construct_prompt(SYSTEM_PROMPT_CORE, prompt, context)

        try:
            logger.info(f"Gemini Thinking... (Depth {depth})")
            response = self.model.generate_content(
                full_text,
                generation_config=genai.types.GenerationConfig(
                    candidate_count=1,
                    max_output_tokens=2048,
                    temperature=0.4
                ),
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            
            text = response.text
            parsed = self._parse_json(text)

            # Tool Handling (Same Logic as HF Client)
            if parsed.get("tool"):
                tool_name = parsed["tool"]
                tool_args = parsed.get("args", {})
                logger.info(f"Gemini requested tool: {tool_name}")
                
                # Execute tool (Import here to avoid circulars)
                tool_out = self._execute_tool(tool_name, tool_args)
                
                # Recursion
                follow_up = f"{prompt}\n[TOOL OUTPUT from {tool_name}]: {json.dumps(tool_out)}\nInterpret this."
                return self.consult(follow_up, context, depth=depth+1)

            return parsed

        except Exception as e:
            logger.error(f"Gemini Error: {e}")
            return {"error": str(e)}

    def _construct_prompt(self, system, user, context):
        c_str = json.dumps(context, indent=2) if context else ""
        return f"""
System: {system}

Context Data:
{c_str}

User Query: {user}

Important: Respond ONLY with valid JSON.
"""

    def _parse_json(self, text):
        text = text.strip()
        if "```json" in text:
            try:
                s = text.index("```json") + 7
                e = text.rindex("```")
                return json.loads(text[s:e])
            except: pass
        if text.startswith("{"):
            try: return json.loads(text)
            except: pass
        return {"analysis": text, "raw_text": True}

    def _execute_tool(self, name, args):
        # Re-use tools implementation
        from .tools import fetch_target_profile, fetch_structure_metadata, fetch_bioactivity, fetch_pockets_for_pdb
        try:
            if name == "fetch_target_profile": return fetch_target_profile(args.get("uniprot_id"))
            if name == "fetch_structure_metadata": return fetch_structure_metadata(args.get("pdb_id"))
            if name == "fetch_bioactivity": return fetch_bioactivity(args.get("chembl_id"))
            if name == "fetch_pockets_for_pdb": return fetch_pockets_for_pdb(args.get("pdb_id"))
            return {"error": "Unknown Tool"}
        except Exception as e: return {"error": str(e)}

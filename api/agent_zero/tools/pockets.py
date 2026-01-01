import requests
import logging
from services.cavity_detector import CavityDetector

logger = logging.getLogger(__name__)

def fetch_pockets_for_pdb(pdb_id: str) -> dict:
    """
    Fetches a PDB file from RCSB and detects binding pockets.
    Returns a list of center/size coordinates for grid boxes.
    """
    if not pdb_id:
        return {"error": "No PDB ID provided"}

    clean_id = pdb_id.strip().upper()
    url = f"https://files.rcsb.org/download/{clean_id}.pdb"

    try:
        # 1. Fetch PDB Content
        response = requests.get(url, timeout=30)
        if response.status_code == 404:
            return {"error": f"PDB ID {clean_id} not found at RCSB."}
        response.raise_for_status()
        pdb_content = response.text
        
        # 2. Run Detection
        detector = CavityDetector()
        cavities = detector.detect_cavities(pdb_content)
        
        if not cavities:
            return {"pdb_id": clean_id, "pockets": [], "message": "No cavities detected."}
            
        # Format for Agent
        formatted_pockets = []
        for c in cavities[:3]: # limit to top 3
            formatted_pockets.append({
                "rank": c['id'],
                "center": c['center'],
                "size": c['size'],
                "score": c.get('score', 0)
            })

        return {
            "pdb_id": clean_id,
            "pocket_count": len(cavities),
            "top_pockets": formatted_pockets,
            "recommendation": f"Use pocket {formatted_pockets[0]['rank']} centered at {formatted_pockets[0]['center']}." if formatted_pockets else "No actionable pockets."
        }

    except Exception as e:
        logger.error(f"Pocket Tool Error: {e}")
        return {"error": f"Failed to detect pockets: {str(e)}"}

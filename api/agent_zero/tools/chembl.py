import requests
import logging

logger = logging.getLogger(__name__)

def fetch_bioactivity(chembl_id: str) -> dict:
    """
    Fetches bioactivity data (IC50, Ki, KD) for a given ChEMBL ID.
    Returns the top 5 most potent activities found.
    """
    if not chembl_id:
        return {"error": "No ChEMBL ID provided"}

    clean_id = chembl_id.strip().upper()
    # Ensure ID format if user just typed number
    if not clean_id.startswith("CHEMBL"):
        return {"error": f"Invalid ID format. Expected 'CHEMBL...', got '{clean_id}'"}

    # https://www.ebi.ac.uk/chembl/api/data/activity
    url = f"https://www.ebi.ac.uk/chembl/api/data/activity.json"
    params = {
        "molecule_chembl_id": clean_id,
        "standard_type__in": "IC50,Ki,Kd,EC50",
        "limit": 5,
        "ordering": "standard_value" # Get most potent (lowest value) first
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        activities = data.get("activities", [])
        
        if not activities:
            return {"chembl_id": clean_id, "bioactivity": "No IC50/Ki/Kd/EC50 data found."}

        results = []
        for act in activities:
            target_chembl_id = act.get("target_chembl_id", "Unknown")
            # Ideally we'd fetch target name too, but let's keep it simple or rely on Agent to infer
            type_ = act.get("standard_type")
            value = act.get("standard_value")
            units = act.get("standard_units")
            relation = act.get("standard_relation", "=")
            
            if value and units:
                results.append(f"{type_} {relation} {value} {units} (Target: {target_chembl_id})")

        return {
            "chembl_id": clean_id,
            "found_activities": len(activities),
            "top_activities": results
        }

    except Exception as e:
        logger.error(f"ChEMBL Fetch Error: {e}")
        return {"error": f"Failed to fetch ChEMBL data: {str(e)}"}

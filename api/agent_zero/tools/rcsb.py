import requests
import logging

logger = logging.getLogger(__name__)

def fetch_structure_metadata(pdb_id: str) -> dict:
    """
    Fetches structure metadata from RCSB PDB.
    Returns experimental method, resolution, and title.
    """
    if not pdb_id:
        return {"error": "No PDB ID provided"}

    clean_id = pdb_id.strip().upper()
    url = f"https://data.rcsb.org/rest/v1/core/entry/{clean_id}"

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 404:
            return {"error": f"PDB ID {clean_id} not found."}
        response.raise_for_status()
        
        data = response.json()
        
        # Extract meaningful fields
        title = data.get("struct", {}).get("title", "Unknown Title")
        exptl = data.get("exptl", [{}])[0]
        method = exptl.get("method", "Unknown Method")
        
        # Resolution often in different places depending on method (X-ray vs NMR vs EM)
        resolution = "N/A"
        if "rcsb_entry_info" in data:
            resolution = data["rcsb_entry_info"].get("resolution_combined", [None])[0]
        
        if not resolution:
             resolution = data.get("refine", [{}])[0].get("ls_d_res_high", "Unknown")

        return {
            "pdb_id": clean_id,
            "title": title,
            "method": method,
            "resolution": resolution,
            "release_date": data.get("rcsb_accession_info", {}).get("initial_release_date", "Unknown"),
            "organism": data.get("rcsb_entity_source_organism", [{}])[0].get("scientific_name", "Unknown")
        }

    except Exception as e:
        logger.error(f"RCSB Fetch Error: {e}")
        return {"error": f"Failed to fetch data: {str(e)}"}

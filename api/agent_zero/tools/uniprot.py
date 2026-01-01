import requests
import logging

logger = logging.getLogger(__name__)

def fetch_target_profile(uniprot_id: str) -> dict:
    """
    Fetches protein target metadata from UniProt.
    Returns details on function, gene names, and active sites.
    """
    if not uniprot_id:
        return {"error": "No UniProt ID provided"}

    clean_id = uniprot_id.strip().upper()
    url = f"https://rest.uniprot.org/uniprotkb/{clean_id}.json"

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 404:
            return {"error": f"UniProt ID {clean_id} not found."}
        response.raise_for_status()
        
        data = response.json()
        
        # Extract meaningful fields
        protein_name = data.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "Unknown")
        
        genes = data.get("genes", [])
        gene_name = genes[0].get("geneName", {}).get("value", "Unknown") if genes else "Unknown"
        
        comments = data.get("comments", [])
        function_comment = next((c['texts'][0]['value'] for c in comments if c['commentType'] == 'FUNCTION'), "No function description available.")
        
        # Features (Active sites, Binding sites)
        features = data.get("features", [])
        active_sites = [f for f in features if f['type'] == 'ACTIVE_SITE']
        binding_sites = [f for f in features if f['type'] == 'BINDING_SITE']
        
        site_details = []
        for site in active_sites + binding_sites:
            desc = site.get("description", "Site")
            loc = site.get("location", {}).get("start", {}).get("value", "?")
            site_details.append(f"{desc} at residue {loc}")

        return {
            "uniprot_id": clean_id,
            "protein_name": protein_name,
            "gene_name": gene_name,
            "function": function_comment[:500] + "..." if len(function_comment) > 500 else function_comment,
            "length": data.get("sequence", {}).get("length", 0),
            "organism": data.get("organism", {}).get("scientificName", "Unknown"),
            "important_sites": site_details
        }

    except Exception as e:
        logger.error(f"UniProt Fetch Error: {e}")
        return {"error": f"Failed to fetch data: {str(e)}"}

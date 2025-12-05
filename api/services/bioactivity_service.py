"""
Bioactivity Database Service
Integrates with PubChem and ChEMBL for compound bioactivity data.
All APIs are FREE and open access.
"""

import requests
import logging
from typing import Dict, List, Optional
import urllib.parse

logger = logging.getLogger(__name__)


class BioactivityService:
    """
    Fetches bioactivity data from public databases.
    - PubChem: Compound properties, bioassays
    - ChEMBL: Bioactive molecules, targets, assays
    """
    
    PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'BioDockify/1.0'
        })
    
    def get_compound_info(self, smiles: str) -> Dict:
        """
        Get comprehensive compound information from multiple sources.
        """
        result = {
            "smiles": smiles,
            "pubchem": self._get_pubchem_data(smiles),
            "chembl": self._get_chembl_data(smiles),
            "external_links": self._get_database_links(smiles)
        }
        return result
    
    def _get_pubchem_data(self, smiles: str) -> Dict:
        """
        Fetch compound data from PubChem.
        """
        try:
            encoded_smiles = urllib.parse.quote(smiles, safe='')
            
            # Get compound CID first
            cid_url = f"{self.PUBCHEM_BASE}/compound/smiles/{encoded_smiles}/cids/JSON"
            cid_response = self.session.get(cid_url, timeout=10)
            
            if cid_response.status_code != 200:
                return {"found": False, "message": "Compound not found in PubChem"}
            
            cid_data = cid_response.json()
            cid = cid_data.get("IdentifierList", {}).get("CID", [None])[0]
            
            if not cid:
                return {"found": False, "message": "No CID found"}
            
            # Get compound properties
            props_url = f"{self.PUBCHEM_BASE}/compound/cid/{cid}/property/MolecularFormula,MolecularWeight,IUPACName,CanonicalSMILES,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount/JSON"
            props_response = self.session.get(props_url, timeout=10)
            
            if props_response.status_code == 200:
                props = props_response.json().get("PropertyTable", {}).get("Properties", [{}])[0]
            else:
                props = {}
            
            # Get synonyms (common names)
            synonyms_url = f"{self.PUBCHEM_BASE}/compound/cid/{cid}/synonyms/JSON"
            syn_response = self.session.get(synonyms_url, timeout=10)
            synonyms = []
            if syn_response.status_code == 200:
                syn_data = syn_response.json()
                synonyms = syn_data.get("InformationList", {}).get("Information", [{}])[0].get("Synonym", [])[:5]
            
            return {
                "found": True,
                "cid": cid,
                "iupac_name": props.get("IUPACName"),
                "molecular_formula": props.get("MolecularFormula"),
                "molecular_weight": props.get("MolecularWeight"),
                "xlogp": props.get("XLogP"),
                "tpsa": props.get("TPSA"),
                "hbd": props.get("HBondDonorCount"),
                "hba": props.get("HBondAcceptorCount"),
                "synonyms": synonyms,
                "url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"
            }
            
        except Exception as e:
            logger.error(f"PubChem API error: {e}")
            return {"found": False, "error": str(e)}
    
    def _get_chembl_data(self, smiles: str) -> Dict:
        """
        Fetch compound data from ChEMBL.
        """
        try:
            # Search by SMILES
            search_url = f"{self.CHEMBL_BASE}/molecule/search.json"
            params = {"q": smiles, "limit": 1}
            
            response = self.session.get(search_url, params=params, timeout=10)
            
            if response.status_code != 200:
                return {"found": False, "message": "ChEMBL search failed"}
            
            data = response.json()
            molecules = data.get("molecules", [])
            
            if not molecules:
                return {"found": False, "message": "Compound not found in ChEMBL"}
            
            mol = molecules[0]
            chembl_id = mol.get("molecule_chembl_id")
            
            # Get bioactivity data
            activities = self._get_chembl_activities(chembl_id) if chembl_id else []
            
            return {
                "found": True,
                "chembl_id": chembl_id,
                "pref_name": mol.get("pref_name"),
                "max_phase": mol.get("max_phase"),  # Drug development phase
                "molecule_type": mol.get("molecule_type"),
                "therapeutic_flag": mol.get("therapeutic_flag"),
                "oral": mol.get("oral"),
                "activities": activities[:10],  # Top 10 activities
                "url": f"https://www.ebi.ac.uk/chembl/compound_report_card/{chembl_id}/" if chembl_id else None
            }
            
        except Exception as e:
            logger.error(f"ChEMBL API error: {e}")
            return {"found": False, "error": str(e)}
    
    def _get_chembl_activities(self, chembl_id: str) -> List[Dict]:
        """
        Get bioactivity data for a ChEMBL compound.
        """
        try:
            url = f"{self.CHEMBL_BASE}/activity.json"
            params = {
                "molecule_chembl_id": chembl_id,
                "limit": 20
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            activities = data.get("activities", [])
            
            # Format activities
            formatted = []
            for act in activities:
                if act.get("standard_value") and act.get("standard_type"):
                    formatted.append({
                        "target_name": act.get("target_pref_name"),
                        "target_organism": act.get("target_organism"),
                        "activity_type": act.get("standard_type"),
                        "value": act.get("standard_value"),
                        "units": act.get("standard_units"),
                        "assay_type": act.get("assay_type")
                    })
            
            return formatted
            
        except Exception as e:
            logger.error(f"ChEMBL activities error: {e}")
            return []
    
    def _get_database_links(self, smiles: str) -> Dict:
        """
        Generate links to external compound databases.
        """
        encoded = urllib.parse.quote(smiles, safe='')
        
        return {
            "pubchem": {
                "name": "PubChem",
                "search_url": f"https://pubchem.ncbi.nlm.nih.gov/#query={encoded}",
                "description": "Compound properties, bioassays, patents"
            },
            "chembl": {
                "name": "ChEMBL",
                "search_url": f"https://www.ebi.ac.uk/chembl/g/#search_results/all/query={encoded}",
                "description": "Bioactive molecules, targets, assays"
            },
            "drugbank": {
                "name": "DrugBank",
                "search_url": f"https://go.drugbank.com/unearth/q?query={encoded}",
                "description": "Drug and target information"
            },
            "zinc": {
                "name": "ZINC",
                "search_url": f"https://zinc.docking.org/substances/search/?q={encoded}",
                "description": "Commercially available compounds"
            },
            "bindingdb": {
                "name": "BindingDB",
                "search_url": f"https://www.bindingdb.org/rwd/bind/chemsearch/marvin/SDFdownload.jsp?download=yes&smiles={encoded}",
                "description": "Protein-ligand binding data"
            }
        }
    
    def get_similar_drugs(self, smiles: str, threshold: float = 0.7) -> List[Dict]:
        """
        Find similar approved drugs using PubChem.
        """
        try:
            encoded = urllib.parse.quote(smiles, safe='')
            url = f"{self.PUBCHEM_BASE}/compound/similarity/smiles/{encoded}/JSON"
            params = {"Threshold": int(threshold * 100), "MaxRecords": 10}
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code != 200:
                return []
            
            # This is an async operation in PubChem
            # For simplicity, return the search URL instead
            return [{
                "message": "Use PubChem similarity search",
                "url": f"https://pubchem.ncbi.nlm.nih.gov/#query={encoded}&input_type=similarity"
            }]
            
        except Exception as e:
            logger.error(f"Similar drugs search error: {e}")
            return []


def get_bioactivity_data(smiles: str) -> Dict:
    """Convenience function to get all bioactivity data."""
    service = BioactivityService()
    return service.get_compound_info(smiles)

from rdkit import Chem
from rdkit.Chem import FilterCatalog
import logging

logger = logging.getLogger(__name__)

class ToxicityEngine:
    def __init__(self):
        self._initialize_filters()

    def _initialize_filters(self):
        """Initialize RDKit filter catalogs for toxicity alerts"""
        # PAINS (Pan Assay Interference Compounds)
        self.pains_params = FilterCatalog.FilterCatalogParams()
        self.pains_params.AddCatalogs(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        self.pains_catalog = FilterCatalog.FilterCatalog(self.pains_params)

        # BRENK (Unwanted substructures)
        self.brenk_params = FilterCatalog.FilterCatalogParams()
        self.brenk_params.AddCatalogs(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
        self.brenk_catalog = FilterCatalog.FilterCatalog(self.brenk_params)

        # NIH (Exclude)
        self.nih_params = FilterCatalog.FilterCatalogParams()
        self.nih_params.AddCatalogs(FilterCatalog.FilterCatalogParams.FilterCatalogs.NIH)
        self.nih_catalog = FilterCatalog.FilterCatalog(self.nih_params)

    def predict(self, smiles_list: list) -> list:
        results = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if not mol:
                    results.append({"smiles": smiles, "valid": False, "error": "Invalid SMILES"})
                    continue

                # Run Checks
                alerts = []
                
                # PAINS Check
                if self.pains_catalog.HasMatch(mol):
                    entries = self.pains_catalog.GetMatches(mol)
                    for entry in entries:
                        alerts.append({"type": "PAINS", "description": entry.GetDescription()})

                # BRENK Check
                if self.brenk_catalog.HasMatch(mol):
                    entries = self.brenk_catalog.GetMatches(mol)
                    for entry in entries:
                        alerts.append({"type": "Unwanted (Brenk)", "description": entry.GetDescription()})

                # NIH Check
                if self.nih_catalog.HasMatch(mol):
                    entries = self.nih_catalog.GetMatches(mol)
                    for entry in entries:
                        alerts.append({"type": "NIH Excluded", "description": entry.GetDescription()})

                # Basic Molecular Props (Lipinski violation check could go here)
                
                # Determine Risk Level
                risk = "Low"
                if len(alerts) > 0:
                    risk = "Moderate"
                if any(a['type'] == 'PAINS' for a in alerts):
                    risk = "High"

                results.append({
                    "smiles": smiles,
                    "valid": True,
                    "risk": risk,
                    "alerts": alerts,
                    "alert_count": len(alerts)
                })

            except Exception as e:
                logger.error(f"Error processing {smiles}: {str(e)}")
                results.append({"smiles": smiles, "valid": False, "error": str(e)})

        return results

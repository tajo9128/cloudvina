import httpx
from typing import List, Dict, Optional
import asyncio

class ChEMBLService:
    """
    Service to interact with the EBI ChEMBL API for target prediction.
    Strategy:
    1. Similarity Search: Search for compounds similar to the input SMILES (>80%).
    2. Get Activities: For the most similar compounds, get their bioactivities.
    3. Get Targets: Identify valid targets (IC50/Ki < 1000nM) and aggregate them.
    4. Rank: Return top targets by frequency and similarity score.
    """
    
    BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
    
    async def predict_targets(self, smiles: str, similarity_cutoff: int = 70) -> List[Dict]:
        """
        Orchestrate the target prediction flow.
        """
        try:
            # 1. Similarity Search
            similar_molecules = await self._search_similar_molecules(smiles, similarity_cutoff)
            if not similar_molecules:
                return []
            
            # 2. Get Targets for these molecules
            targets = await self._get_targets_for_molecules(similar_molecules)
            
            # 3. Format and Rank
            ranked_targets = self._rank_targets(targets)
            
            return ranked_targets
            
        except Exception as e:
            print(f"[ChEMBL Error] Target prediction failed: {str(e)}")
            return []

    async def _search_similar_molecules(self, smiles: str, cutoff: int) -> List[Dict]:
        """
        Query ChEMBL's similarity endpoint.
        Returns: List of {molecule_chembl_id, similarity}
        """
        url = f"{self.BASE_URL}/similarity/{smiles}/{cutoff}?format=json"
        
        async with httpx.AsyncClient() as client:
            try:
                # Timeout increased as ChEMBL similarity search can be slow
                resp = await client.get(url, timeout=30.0) 
                if resp.status_code != 200:
                    return []
                
                data = resp.json()
                molecules = data.get('molecules', [])
                
                # Sort by similarity desc and take top 10
                molecules.sort(key=lambda x: float(x['similarity']), reverse=True)
                return molecules[:10]
                
            except Exception as e:
                print(f"[ChEMBL Error] Similarity search failed: {e}")
                return []

    async def _get_targets_for_molecules(self, molecules: List[Dict]) -> List[Dict]:
        """
        For each molecule, fetch its bioactivities and associated targets.
        """
        all_targets = []
        
        async with httpx.AsyncClient() as client:
            tasks = []
            for mol in molecules:
                chembl_id = mol['molecule_chembl_id']
                similarity = float(mol['similarity'])
                tasks.append(self._fetch_molecule_activities(client, chembl_id, similarity))
            
            # Run parallel requests
            results = await asyncio.gather(*tasks)
            
            # Flatten results
            for res in results:
                all_targets.extend(res)
                
        return all_targets

    async def _fetch_molecule_activities(self, client: httpx.AsyncClient, chembl_id: str, similarity: float) -> List[Dict]:
        """
        Fetch High-Confidence activities (IC50/Ki/EC50) for a molecule.
        """
        # Filter for relevant activity types and low values (potent binding)
        # Note: ChEMBL API filtering via URL params
        # pChEMBL_value > 6 means < 1uM potency (pChEMBL = -log10(Molar))
        url = (
            f"{self.BASE_URL}/activity.json"
            f"?molecule_chembl_id={chembl_id}"
            f"&pchembl_value__gte=6" 
            f"&limit=5"
        )
        
        try:
            resp = await client.get(url, timeout=10.0)
            if resp.status_code != 200:
                return []
            
            activities = resp.json().get('activities', [])
            targets = []
            
            for act in activities:
                # We only care about human targets for now, generally
                # But ChEMBL returns everything. We'll filter later or accept all.
                target_chembl_id = act.get('target_chembl_id')
                target_name = act.get('target_pref_name')
                
                if target_chembl_id and target_name:
                    targets.append({
                        'chembl_id': target_chembl_id,
                        'target_name': target_name,
                        'organism': act.get('target_organism', 'Unknown'),
                        'source_similarity': similarity,
                        'activity_type': act.get('standard_type'),
                        'activity_value': act.get('standard_value'),
                        'molecule_chembl_id': chembl_id
                    })
            return targets
            
        except Exception as e:
            # print(f"Activity fetch error for {chembl_id}: {e}")
            return []

    def _rank_targets(self, raw_targets: List[Dict]) -> List[Dict]:
        """
        Aggregate targets by ID and calculate a 'probability' score.
        Score based on:
        - Frequency of appearance across similar molecules
        - Similarity of the source molecules
        """
        aggregated = {}
        
        for t in raw_targets:
            tid = t['chembl_id']
            if tid not in aggregated:
                aggregated[tid] = {
                    'target': t['target_name'],
                    'common_name': t['target_name'], # Use pref_name as common
                    'uniprot_id': tid, # Placeholder, as we have ChEMBL ID
                    'organism': t['organism'],
                    'hits': 0,
                    'total_similarity': 0,
                    'max_similarity': 0
                }
            
            agg = aggregated[tid]
            agg['hits'] += 1
            agg['total_similarity'] += t['source_similarity']
            agg['max_similarity'] = max(agg['max_similarity'], t['source_similarity'])
        
        # Calculate final score
        results = []
        for tid, data in aggregated.items():
            # A simple heuristic score:
            # Base it heavily on the MAX similarity of the ligand that hit it
            # Bonus for multiple hits
            
            # Normalize similarity (0-100 -> 0-1)
            base_score = data['max_similarity'] / 100.0
            
            # Frequency bonus: +5% per extra hit, max +20%
            freq_bonus = min(0.2, (data['hits'] - 1) * 0.05)
            
            final_prob = min(0.99, base_score + freq_bonus)
            
            # Filter non-human if specific requirements (Optional)
            # For now keep all
            
            results.append({
                "target": data['target'],
                "common_name": data['organism'], # Show organism in common name field
                "uniprot_id": tid, # Returning ChEMBL ID as UniProt ID is harder to fetch
                "probability": round(final_prob, 2)
            })
            
        # Sort by probability desc
        results.sort(key=lambda x: x['probability'], reverse=True)
        
        return results[:10]  # Top 10

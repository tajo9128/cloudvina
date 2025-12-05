import random
try:
    import oddt
    from oddt.scoring import rfscore
except ImportError:
    oddt = None
    rfscore = None
    print("Warning: ODDT not found. Evolution will use RDKit scoring only.")
from rdkit import Chem
from rdkit.Chem import AllChem
# from rdkit.Chem import rdMolChem
from rdkit.Chem import QED, Crippen
import numpy as np
import io
import os

import tempfile
import os

class GeneticAlgorithm:
    def __init__(self, receptor_pdbqt_content, center, size):
        """
        Initialize the Evolutionary Engine.
        """
        self.receptor_content = receptor_pdbqt_content
        self.center = center
        self.size = size
        self.protein = None
        self.temp_receptor_path = None
        
        # Load ODDT Scoring Function (RF-Score v1)
        try:
            self.scorer = rfscore.rfscore(version=1)
            
            # Save receptor to temp file for ODDT
            self.temp_receptor = tempfile.NamedTemporaryFile(delete=False, suffix='.pdbqt')
            self.temp_receptor.write(receptor_pdbqt_content.encode('utf-8'))
            self.temp_receptor.close()
            self.temp_receptor_path = self.temp_receptor.name
            
            # Load Protein into ODDT
            # Note: ODDT reads files based on extension
            self.protein = next(oddt.toolkit.readfile('pdbqt', self.temp_receptor_path))
            self.protein.protein = True
            
            # Set the scorer's protein
            self.scorer.set_protein(self.protein)
            
        except Exception as e:
            print(f"Warning: Could not load ODDT/Receptor: {e}")
            self.scorer = None
            self.protein = None
        
        # Population parameters
        self.population_size = 50
        self.mutation_rate = 0.3
        self.crossover_rate = 0.5
        self.generations = 10

    def __del__(self):
        """Cleanup temp files"""
        if self.temp_receptor_path and os.path.exists(self.temp_receptor_path):
            try:
                os.unlink(self.temp_receptor_path)
            except:
                pass

    # ... (rest of the class) ...

    def score_molecule(self, mol):
        """
        Calculate the 'Fitness' of a molecule.
        Primary: ODDT RF-Score (Machine Learning).
        Secondary: RDKit Descriptors (QED + LogP) as fallback/penalty.
        """
        try:
            # 1. RDKit Descriptors (Drug-likeness)
            qed_score = Chem.QED.qed(mol)
            logp = Chem.Crippen.MolLogP(mol)
            
            # LogP Penalty (Target 0-5)
            logp_penalty = 0
            if logp < 0: logp_penalty = abs(logp)
            elif logp > 5: logp_penalty = logp - 5
            
            # Base score from chemical properties (max ~ -10)
            base_score = -(qed_score * 10.0) + (logp_penalty * 0.5)
            
            # 2. ODDT Scoring (if available)
            if self.scorer and self.protein:
                try:
                    # Convert RDKit Mol to ODDT Mol
                    oddt_mol = oddt.toolkit.Molecule(mol)
                    
                    # Predict Binding Affinity (pKd or similar)
                    # RF-Score returns an array, we take the first value
                    # Higher is better for pKd (usually)
                    # But we want to minimize energy, so we might need to invert or check metric
                    # RF-Score v1 predicts pKd (affinity). Higher is better.
                    # We want to maximize this.
                    
                    affinity = self.scorer.predict(oddt_mol)[0]
                    
                    # Combine: Affinity (e.g., 6.5) + DrugLikeness (e.g., -5.0)
                    # We want to MAXIMIZE the total score
                    final_score = (affinity * 2.0) + base_score
                    return final_score
                    
                except Exception as e:
                    print(f"ODDT Scoring Error: {e}")
            
            return base_score
            
        except Exception as e:
            print(f"Scoring error: {e}")
            return 0.0
        
    def initialize_population(self, seed_smiles_list):
        """Create initial population from seed molecules."""
        population = []
        for smiles in seed_smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol) # Generate 3D coords
                population.append(mol)
        
        # Fill the rest with random mutations of the seeds
        while len(population) < self.population_size:
            parent = random.choice(population[:len(seed_smiles_list)])
            child = self.mutate(parent)
            if child:
                population.append(child)
                
        return population

    def evolve(self, seed_smiles_list):
        """
        Run the evolution loop.
        Yields the best molecule of each generation.
        """
        population = self.initialize_population(seed_smiles_list)
        
        for generation in range(1, self.generations + 1):
            # 1. Score Population
            scored_population = []
            for mol in population:
                score = self.score_molecule(mol)
                scored_population.append((score, mol))
            
            # Sort by score (lower is better for binding energy)
            scored_population.sort(key=lambda x: x[0])
            
            # Yield Best of Generation
            best_score, best_mol = scored_population[0]
            yield {
                "generation": generation,
                "best_score": best_score,
                "sdf": Chem.MolToMolBlock(best_mol)
            }
            
            # 2. Selection (Elitism + Tournament)
            # Keep top 10%
            elite_count = int(self.population_size * 0.1)
            new_population = [mol for score, mol in scored_population[:elite_count]]
            
            # 3. Breeding
            while len(new_population) < self.population_size:
                if random.random() < self.crossover_rate:
                    parent1 = self.tournament_select(scored_population)
                    parent2 = self.tournament_select(scored_population)
                    child = self.crossover(parent1, parent2)
                else:
                    parent = self.tournament_select(scored_population)
                    child = self.mutate(parent)
                
                if child:
                     # Basic validity check
                    try:
                        Chem.SanitizeMol(child)
                        AllChem.EmbedMolecule(child)
                        new_population.append(child)
                    except:
                        pass # Discard invalid molecules
            
            population = new_population

    def tournament_select(self, scored_population):
        """Select a parent using tournament selection."""
        tournament_size = 3
        tournament = random.sample(scored_population, tournament_size)
        tournament.sort(key=lambda x: x[0])
        return tournament[0][1]

    def mutate(self, mol):
        """
        Apply a random chemical mutation using RDKit.
        """
        try:
            # Clone the molecule
            new_mol = Chem.RWMol(mol)
            
            mutation_type = random.choice(['add_atom', 'remove_atom', 'change_bond'])
            
            if mutation_type == 'add_atom':
                # Add a Carbon to a random atom
                if new_mol.GetNumAtoms() > 0:
                    idx = random.randint(0, new_mol.GetNumAtoms() - 1)
                    new_atom = Chem.Atom(6) # Carbon
                    new_idx = new_mol.AddAtom(new_atom)
                    new_mol.AddBond(idx, new_idx, Chem.BondType.SINGLE)
            
            elif mutation_type == 'remove_atom':
                # Remove a random atom (if not the only one)
                if new_mol.GetNumAtoms() > 1:
                    idx = random.randint(0, new_mol.GetNumAtoms() - 1)
                    new_mol.RemoveAtom(idx)
                    
            elif mutation_type == 'change_bond':
                # Change a bond order
                if new_mol.GetNumBonds() > 0:
                    bonds = new_mol.GetBonds()
                    bond = random.choice(bonds)
                    current_order = bond.GetBondType()
                    new_order = Chem.BondType.DOUBLE if current_order == Chem.BondType.SINGLE else Chem.BondType.SINGLE
                    bond.SetBondType(new_order)

            # Sanitize to ensure chemical validity
            Chem.SanitizeMol(new_mol)
            return new_mol.GetMol()
            
        except Exception:
            return None

    def crossover(self, mol1, mol2):
        """
        Combine two molecules. 
        For now, just return one of them mutated (placeholder for real fragment crossover).
        """
        return self.mutate(mol1)

    def score_molecule(self, mol):
        """
        Calculate the 'Fitness' of a molecule.
        Currently uses RDKit Descriptors (QED + LogP) as a proxy for 'Drug-likeness'.
        In the future, this will call Vina for binding affinity.
        """
        try:
            # 1. Calculate QED (Quantitative Estimation of Drug-likeness) - Range [0, 1]
            qed_score = Chem.QED.qed(mol)
            
            # 2. Calculate LogP (Octanol-Water Partition Coefficient) - Target range 0-5
            logp = Chem.Crippen.MolLogP(mol)
            
            # Penalty for LogP being outside optimal range (0-5)
            logp_penalty = 0
            if logp < 0: logp_penalty = abs(logp)
            elif logp > 5: logp_penalty = logp - 5
            
            # 3. Synthetic Accessibility (SA) - Optional, but good for realism
            # For now, we stick to QED and LogP
            
            # Final Score formulation (We want to MAXIMIZE this for the GA, 
            # but usually Binding Energy is minimized. Let's return a negative value to mimic Energy)
            # Higher QED = More negative (better) "Pseudo-Energy"
            
            # Formula: -10 * QED + LogP_Penalty
            # A perfect drug (QED=1, LogP=3) would be around -10.0
            
            pseudo_energy = -(qed_score * 10.0) + (logp_penalty * 0.5)
            
            return pseudo_energy
            
        except Exception as e:
            print(f"Scoring error: {e}")
            return 0.0

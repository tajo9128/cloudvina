"""
Trajectory analysis module for BioDockify
Extracts RMSD, RMSF, binding interactions, stability metrics
"""

import MDAnalysis as mda
from MDAnalysis.analysis import rms, contacts
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any


class TrajectoryAnalyzer:
    """Analyze OpenMM-generated trajectories"""
    
    def __init__(self, trajectory_file: str, topology_file: str):
        """
        Args:
            trajectory_file: DCD or XTC file from OpenMM
            topology_file: PDB or GRO file with topology
        """
        self.u = mda.Universe(topology_file, trajectory_file)
        self.protein = self.u.select_atoms('protein')
        self.ligand = self.u.select_atoms('resname LIG')  # Adjust residue name as needed
        self.results = {}
    
    def calculate_rmsd(self, reference_frame: int = 0) -> pd.DataFrame:
        """
        Calculate RMSD of protein backbone over trajectory
        
        Args:
            reference_frame: Frame to use as reference (default: first frame)
            
        Returns:
            DataFrame with time (ns) and RMSD (Å)
        """
        ca = self.u.select_atoms('protein and name CA')
        
        # Align to reference
        aligner = rms.RMSD(
            self.u,
            select='protein and name CA',
            ref_frame=reference_frame
        )
        aligner.run()
        
        # Extract results
        time = aligner.results.rmsd.T[1] / 1000  # Convert to ns
        rmsd_values = aligner.results.rmsd.T[2]  # Backbone RMSD
        
        df = pd.DataFrame({
            'time_ns': time,
            'rmsd_angstrom': rmsd_values
        })
        
        self.results['rmsd'] = df
        return df
    
    def calculate_rmsf(self) -> pd.DataFrame:
        """
        Calculate root-mean-square fluctuation per residue
        
        Returns:
            DataFrame with residue and RMSF
        """
        ca = self.u.select_atoms('protein and name CA')
        rmsf_analyzer = rms.RMSF(ca).run()
        
        df = pd.DataFrame({
            'residue_id': ca.resids,
            'residue_name': ca.resnames,
            'rmsf_angstrom': rmsf_analyzer.results.rmsf
        })
        
        self.results['rmsf'] = df
        return df
    
    def calculate_radius_of_gyration(self) -> pd.DataFrame:
        """
        Calculate radius of gyration (measure of protein compactness)
        
        Returns:
            DataFrame with time and Rg values
        """
        rg_values = []
        time_values = []
        
        for ts in self.u.trajectory:
            # Center of mass
            com = self.protein.center_of_mass()
            # Distances from COM
            distances = np.linalg.norm(self.protein.positions - com, axis=1)
            # RG
            rg = np.sqrt(np.mean(distances ** 2))
            rg_values.append(rg)
            time_values.append(ts.time / 1000)  # Convert to ns
        
        df = pd.DataFrame({
            'time_ns': time_values,
            'radius_of_gyration_angstrom': rg_values
        })
        
        self.results['radius_of_gyration'] = df
        return df
    
    def calculate_ligand_rmsd(self, reference_frame: int = 0) -> pd.DataFrame:
        """
        Calculate RMSD of ligand throughout trajectory
        Important for binding stability assessment
        
        Args:
            reference_frame: Frame to use as reference
            
        Returns:
            DataFrame with time and ligand RMSD
        """
        if len(self.ligand) == 0:
            raise ValueError("No ligand found in trajectory")
        
        # Align to reference
        aligner = rms.RMSD(
            self.u,
            select='resname LIG',  # Ligand
            ref_frame=reference_frame
        )
        aligner.run()
        
        time = aligner.results.rmsd.T[1] / 1000
        rmsd_values = aligner.results.rmsd.T[2]
        
        df = pd.DataFrame({
            'time_ns': time,
            'ligand_rmsd_angstrom': rmsd_values
        })
        
        self.results['ligand_rmsd'] = df
        return df
    
    def calculate_binding_distance(self) -> pd.DataFrame:
        """
        Calculate minimum distance between protein and ligand
        Short distances indicate maintained contact
        
        Returns:
            DataFrame with time and minimum distance
        """
        distances = []
        time_values = []
        
        for ts in self.u.trajectory:
            # Minimum distance between protein and ligand
            dist = contacts.distance_array(
                self.protein.positions,
                self.ligand.positions
            ).min()
            distances.append(dist)
            time_values.append(ts.time / 1000)
        
        df = pd.DataFrame({
            'time_ns': time_values,
            'min_protein_ligand_distance_angstrom': distances
        })
        
        self.results['binding_distance'] = df
        return df
    
    def calculate_hydration_shell(self, distance_cutoff: float = 3.5) -> pd.DataFrame:
        """
        Calculate number of water molecules in hydration shell around ligand
        
        Args:
            distance_cutoff: Distance cutoff in Angstroms
            
        Returns:
            DataFrame with time and water count
        """
        waters = self.u.select_atoms('resname WAT or resname HOH or resname TIP3')
        counts = []
        time_values = []
        
        if len(waters) == 0:
            # No water in system (vacuum simulation)
            return pd.DataFrame({
                'time_ns': [],
                'hydration_shell_waters': []
            })
        
        for ts in self.u.trajectory:
            # Waters within cutoff of ligand COM
            ligand_com = self.ligand.center_of_mass()
            distances = np.linalg.norm(
                waters.positions - ligand_com,
                axis=1
            )
            count = np.sum(distances < distance_cutoff)
            counts.append(count)
            time_values.append(ts.time / 1000)
        
        df = pd.DataFrame({
            'time_ns': time_values,
            'hydration_shell_waters': counts
        })
        
        self.results['hydration_shell'] = df
        return df
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """Run all analyses and return summary"""
        print("Calculating RMSD...")
        self.calculate_rmsd()
        print("Calculating RMSF...")
        self.calculate_rmsf()
        print("Calculating radius of gyration...")
        self.calculate_radius_of_gyration()
        
        # Only calculate ligand metrics if ligand exists
        if len(self.ligand) > 0:
            print("Calculating ligand RMSD...")
            self.calculate_ligand_rmsd()
            print("Calculating binding distance...")
            self.calculate_binding_distance()
            print("Calculating hydration shell...")
            self.calculate_hydration_shell()
        
        return self.get_summary()
    
    def get_summary(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        summary = {}
        
        if 'rmsd' in self.results:
            rmsd_data = self.results['rmsd']['rmsd_angstrom']
            summary['protein_rmsd_mean'] = float(rmsd_data.mean())
            summary['protein_rmsd_std'] = float(rmsd_data.std())
            summary['protein_rmsd_max'] = float(rmsd_data.max())
        
        if 'rmsf' in self.results:
            rmsf_data = self.results['rmsf']['rmsf_angstrom']
            summary['mean_residue_flexibility'] = float(rmsf_data.mean())
            summary['flexible_residues'] = int(np.sum(rmsf_data > 2.0))
        
        if 'radius_of_gyration' in self.results:
            rg_data = self.results['radius_of_gyration']['radius_of_gyration_angstrom']
            summary['rg_mean'] = float(rg_data.mean())
            summary['rg_std'] = float(rg_data.std())
        
        if 'ligand_rmsd' in self.results:
            lig_rmsd = self.results['ligand_rmsd']['ligand_rmsd_angstrom']
            summary['ligand_rmsd_mean'] = float(lig_rmsd.mean())
            summary['ligand_stability'] = 'stable' if lig_rmsd.max() < 3.0 else 'unstable'
        
        if 'binding_distance' in self.results:
            bd = self.results['binding_distance']['min_protein_ligand_distance_angstrom']
            summary['min_binding_distance'] = float(bd.min())
            summary['binding_maintained'] = 'yes' if bd.mean() < 5.0 else 'no'
        
        summary['total_frames'] = len(self.u.trajectory)
        
        return summary
    
    def export_results(self, output_dir: str) -> str:
        """
        Export all analysis results to CSV files
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            Path to output directory
        """
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        
        for name, df in self.results.items():
            df.to_csv(f"{output_dir}/{name}.csv", index=False)
        
        # Export summary
        summary = self.get_summary()
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(f"{output_dir}/summary.csv", index=False)
        
        return output_dir

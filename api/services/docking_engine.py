import os
import sys
import subprocess
import logging
from typing import Dict, Optional

# Import helper services
# These must be copied to /app/services in Docker
try:
    from services.rf_model_service import RFModelService
except ImportError:
    # Handle case where service isn't available in local test env
    RFModelService = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DockingEngine")

class DockingEngine:
    """
    Unified Docking Interface.
    Invokes Vina, Gnina, and RF Model for Consensus Scoring.
    """
    
    def __init__(self, engine_type: str = 'vina'):
        self.engine_type = engine_type.lower()
        
        # Binary paths (Standard in Docker)
        self.vina_bin = '/usr/local/bin/vina'
        self.rdock_bin = '/usr/local/bin/rbdock'
        self.rbcavity_bin = '/usr/local/bin/rbcavity'

    def run_docking(self, receptor_path: str, ligand_path: str, output_path: str, config: Dict = None) -> Dict:
        """
        Run docking workflow.
        """
        params = config or {}
        
        if self.engine_type == 'vina':
            return self._run_vina(receptor_path, ligand_path, output_path, params)
        elif self.engine_type == 'rdock':
            return self._run_rdock(receptor_path, ligand_path, output_path, params)
        elif self.engine_type == 'gnina':
            return self._run_gnina(receptor_path, ligand_path, output_path, params)
        elif self.engine_type == 'consensus':
            return self._run_consensus(receptor_path, ligand_path, output_path, params)
        else:
            raise ValueError(f"Unsupported docking engine: {self.engine_type}")

    def _run_consensus(self, receptor: str, ligand: str, output: str, params: Dict) -> Dict:
        """
        Run Consensus Docking (Vina + Gnina + Random Forest).
        Returns aggregated results.
        """
        logger.info("Starting CONSENSUS Docking Mode...")
        results = {
            "consensus": True,
            "engines": {},
            "best_affinity": 0.0
        }
        
        base_dir = os.path.dirname(output)
        base_name = os.path.splitext(os.path.basename(output))[0]
        
        # 1. Run Vina (Physics-based)
        try:
            out_vina = os.path.join(base_dir, f"{base_name}_vina.pdbqt")
            res_vina = self._run_vina(receptor, ligand, out_vina, params)
            results["engines"]["vina"] = res_vina
        except Exception as e:
            logger.error(f"Consensus: Vina failed: {e}")
            results["engines"]["vina"] = {"error": str(e)}

        # 2. Run Gnina (Deep Learning)
        try:
            out_gnina = os.path.join(base_dir, f"{base_name}_gnina.pdbqt")
            res_gnina = self._run_gnina(receptor, ligand, out_gnina, params)
            results["engines"]["gnina"] = res_gnina
        except Exception as e:
            logger.error(f"Consensus: Gnina failed: {e}")
            results["engines"]["gnina"] = {"error": str(e)}

        # 3. Run RF Model (Machine Learning Rescoring)
        rf_pkd = None
        try:
            if RFModelService:
                # We typically rescore the BEST pose. 
                # Ideally, rescores the output of Vina or Gnina.
                # Here we rescore the best Vina pose (out_vina) against receptor.
                target_ligand = results["engines"].get("vina", {}).get("output_file")
                if target_ligand and os.path.exists(target_ligand):
                    rf_pkd = RFModelService.predict_ligand(receptor, target_ligand)
                    results["engines"]["rf"] = {"pKd": rf_pkd}
                    logger.info(f"RF Model Prediction: {rf_pkd} pKd")
                else:
                    logger.warning("Skipping RF: Vina output not available for rescoring")
            else:
                logger.warning("Skipping RF: Service not imported")
        except Exception as e:
            logger.error(f"Consensus: RF failed: {e}")
            results["engines"]["rf"] = {"error": str(e)}

        # Aggregation Logic
        vina_aff = results["engines"].get("vina", {}).get("best_affinity", 0.0)
        gnina_aff = results["engines"].get("gnina", {}).get("best_affinity", 0.0)
        
        # Calculate Weighted Score (0-10) using ConsensusScorer logic if available locally
        # Otherwise simple aggregation for reporting
        
        # Primary output file: Vina (Best Pose)
        vina_out = results["engines"].get("vina", {}).get("output_file")
        
        if vina_out and os.path.exists(vina_out):
            try:
                with open(vina_out, "r+") as f:
                    content = f.read()
                    f.seek(0, 0)
                    f.write("REMARK 200 ========================================================\n")
                    f.write(f"REMARK 200 CONSENSUS DOCKING RESULTS (Tri-Score)\n")
                    f.write(f"REMARK 200 Vina (Physics):     {vina_aff:.2f} kcal/mol\n")
                    f.write(f"REMARK 200 Gnina (CNN):        {gnina_aff:.2f} kcal/mol\n")
                    f.write(f"REMARK 200 Random Forest (ML): {rf_pkd if rf_pkd else 'N/A'} pKd\n")
                    f.write("REMARK 200 ========================================================\n")
                    f.write(content) 
            except Exception as e:
                logger.error(f"Failed to append consensus remarks: {e}")

        results["output_file"] = vina_out
        
        return results

    def _run_vina(self, receptor: str, ligand: str, output: str, params: Dict) -> Dict:
        """Run AutoDock Vina"""
        # Convert receptor PDB to PDBQT if needed
        if receptor.endswith('.pdb'):
            receptor_pdbqt = receptor.replace('.pdb', '.pdbqt')
            clean_pdb = receptor.replace('.pdb', '_clean.pdb')
            subprocess.run(f"grep -v '^HEADER\|^REMARK\|^AUTHOR\|^REVDAT\|^JRNL' '{receptor}' > '{clean_pdb}'", shell=True, check=True)
            subprocess.run(['obabel', clean_pdb, '-O', receptor_pdbqt, '-xr'], check=True)
            receptor = receptor_pdbqt
        
        cmd = [
            self.vina_bin,
            '--receptor', receptor,
            '--ligand', ligand,
            '--out', output,
            '--center_x', str(params.get('center_x', 0)),
            '--center_y', str(params.get('center_y', 0)),
            '--center_z', str(params.get('center_z', 0)),
            '--size_x', str(params.get('size_x', 20)),
            '--size_y', str(params.get('size_y', 20)),
            '--size_z', str(params.get('size_z', 20)),
            '--cpu', '1',
            '--exhaustiveness', str(params.get('exhaustiveness', 8))
        ]
        
        logger.info(f"Running Vina: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Vina failed: {result.stderr}")
        
        parsed = self._parse_vina_like_output(output, "vina")
        parsed['stdout'] = result.stdout
        parsed['stderr'] = result.stderr
        parsed['command'] = ' '.join(cmd)
        
        try:
            from services.vina_parser import parse_vina_log
            log_parsed = parse_vina_log(result.stdout)
            if log_parsed and 'poses' in log_parsed:
                parsed['poses'] = log_parsed['poses']
        except Exception as e:
            logger.warning(f"Failed to parse Vina log for poses: {e}")
        
        return parsed

    def _run_gnina(self, receptor: str, ligand: str, output: str, params: Dict) -> Dict:
        """Run Gnina (AI Docking)"""
        if receptor.endswith('.pdb'):
            receptor_pdbqt = receptor.replace('.pdb', '.pdbqt')
            clean_pdb = receptor.replace('.pdb', '_clean.pdb')
            subprocess.run(f"grep -v '^HEADER\|^REMARK\|^AUTHOR\|^REVDAT\|^JRNL' '{receptor}' > '{clean_pdb}'", shell=True, check=True)
            subprocess.run(['obabel', clean_pdb, '-O', receptor_pdbqt, '-xr'], check=True)
            receptor = receptor_pdbqt
        
        cmd = [
            '/usr/local/bin/gnina',
            '--receptor', receptor,
            '--ligand', ligand,
            '--out', output,
            '--center_x', str(params.get('center_x', 0)),
            '--center_y', str(params.get('center_y', 0)),
            '--center_z', str(params.get('center_z', 0)),
            '--size_x', str(params.get('size_x', 20)),
            '--size_y', str(params.get('size_y', 20)),
            '--size_z', str(params.get('size_z', 20)),
            '--cpu', '1',
            '--exhaustiveness', '4',
            '--num_modes', '3'
        ]
        
        logger.info(f"Running Gnina: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            logger.error("Gnina timed out after 300 seconds")
            raise Exception("Gnina execution timed out after 5 minutes")
        
        if result.returncode != 0:
            raise Exception(f"Gnina failed: {result.stderr}")
            
        parsed = self._parse_vina_like_output(output, "gnina")
        parsed['stdout'] = result.stdout
        parsed['stderr'] = result.stderr
        parsed['command'] = ' '.join(cmd)
        
        try:
            from services.vina_parser import parse_vina_log
            log_parsed = parse_vina_log(result.stdout)
            if log_parsed and 'poses' in log_parsed:
                parsed['poses'] = log_parsed['poses']
        except Exception as e:
            logger.warning(f"Failed to parse Gnina log for poses: {e}")
            
        return parsed

    def _parse_vina_like_output(self, output_path: str, engine_name: str) -> Dict:
        """Parse PDBQT output from Vina or Gnina"""
        best_affinity = 0.0
        best_cnn_score = None
        
        try:
            with open(output_path, 'r') as f:
                for line in f:
                    if engine_name == 'vina' and line.startswith('REMARK VINA RESULT'):
                        parts = line.split()
                        if len(parts) >= 4:
                            best_affinity = float(parts[3])
                            break
                    
                    if engine_name == 'gnina' and 'minimizedAffinity' in line:
                        parts = line.split()
                        if len(parts) >= 3:
                            best_affinity = float(parts[2])
                    
                    if engine_name == 'gnina' and 'CNNscore' in line:
                        parts = line.split()
                        if len(parts) >= 3:
                            best_cnn_score = float(parts[2])
                            if best_affinity != 0.0:
                                break
                                
        except FileNotFoundError:
             raise Exception(f"{engine_name} output file not found")

        result = {
            "best_affinity": best_affinity,
            "output_file": output_path,
            "engine": engine_name
        }
        
        if best_cnn_score:
            result['cnn_score'] = best_cnn_score
            
        return result

    def _run_rdock(self, receptor_pdbqt: str, ligand_pdbqt: str, output_sdf: str, params: Dict) -> Dict:
        # Placeholder for rDock implementation if needed in future
        raise NotImplementedError("rDock not active in Consensus Mode")

def run_docking_job(engine, receptor, ligand, output, config=None):
    engine = DockingEngine(engine)
    return engine.run_docking(receptor, ligand, output, config)

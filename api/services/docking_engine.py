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

try:
    from services.sidechain_minimizer import SideChainMinimizer
except ImportError:
    SideChainMinimizer = None

try:
    from config.scoring_tiers import ScoringTiers
except ImportError:
    # If config not in path (e.g. Docker issue), fallback to default classStub?
    # Better to assume it works if we installed it.
    # But for safety in this environment:
    from config.scoring_tiers import ScoringTiers


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
        Run Consensus Docking (Smart Funnel).
        Sequence: Vina -> RF -> Gate -> (Gnina?) -> (Minimization?)
        """
        logger.info("Starting CONSENSUS Docking Mode (Smart Funnel)...")
        results = {
            "consensus": True,
            "engines": {},
            "best_affinity": 0.0,
            "analysis_depth": "Unknown"
        }
        
        base_dir = os.path.dirname(output)
        base_name = os.path.splitext(os.path.basename(output))[0]
        
        # 1. Run Vina (Physics-based) - ALWAYS RUNS
        try:
            out_vina = os.path.join(base_dir, f"{base_name}_vina.pdbqt")
            res_vina = self._run_vina(receptor, ligand, out_vina, params)
            results["engines"]["vina"] = res_vina
            vina_aff = res_vina.get("best_affinity", 0.0)
        except Exception as e:
            logger.error(f"Consensus: Vina failed: {e}")
            results["engines"]["vina"] = {"error": str(e)}
            return results # Critical failure

        # 2. Run RF Score (Fast ML Filter) - ALWAYS RUNS
        rf_pkd = 0.0
        try:
            if RFModelService:
                # Rescore Vina output
                if out_vina and os.path.exists(out_vina):
                    rf_pkd = RFModelService.predict_ligand(receptor, out_vina)
                    results["engines"]["rf"] = {"pKd": rf_pkd}
                    logger.info(f"RF Model Prediction: {rf_pkd} pKd")
                else:
                    logger.warning("Skipping RF: Vina output missing")
            else:
                logger.warning("Skipping RF: Service not imported")
        except Exception as e:
            logger.error(f"Consensus: RF failed: {e}")
            results["engines"]["rf"] = {"error": str(e)}

        # 3. Gating Logic (Smart Funnel)
        tier_info = ScoringTiers.get_tier(rf_score=rf_pkd, vina_affinity=vina_aff)
        results["tier_info"] = tier_info
        logger.info(f"Ligand Tier: {tier_info['label']} ({tier_info['tier']})")
        if tier_info.get('is_anomaly'):
            logger.warning(f"Anomaly Detected: {tier_info.get('anomaly_reason')}")

        # 4. Run Gnina (Deep Learning) - CONDITIONAL
        gnina_aff = 0.0
        if tier_info.get('gnina_enabled', True): # Default True if tiering fails
            try:
                out_gnina = os.path.join(base_dir, f"{base_name}_gnina.pdbqt")
                # Note: We could seed Gnina with Vina pose to speed it up further, 
                # but for independence we keep standard run or user defined params.
                res_gnina = self._run_gnina(receptor, ligand, out_gnina, params)
                results["engines"]["gnina"] = res_gnina
                gnina_aff = res_gnina.get("best_affinity", 0.0)
            except Exception as e:
                logger.error(f"Consensus: Gnina failed: {e}")
                results["engines"]["gnina"] = {"error": str(e)}
        else:
            logger.info("Skipping Gnina (Filtered by Tier)")
            results["engines"]["gnina"] = {"skipped": True, "reason": tier_info['tier']}

        # 5. Side-Chain Minimization - CONDITIONAL
        minimized_struct = None
        if tier_info.get('minimization_enabled', False) and SideChainMinimizer:
             try:
                logger.info(f"Triggering Side-Chain Minimization...")
                # Prefer Gnina output if available (refined), else Vina
                target_ligand = results["engines"].get("gnina", {}).get("output_file")
                if not target_ligand or not os.path.exists(target_ligand):
                     target_ligand = out_vina
                
                minimizer = SideChainMinimizer(receptor, target_ligand)
                min_dir = os.path.join(base_dir, "minimized")
                rel_prot, rel_lig = minimizer.minimize(output_dir=min_dir)
                
                results["minimized"] = True
                results["minimized_receptor"] = rel_prot
                results["minimized_ligand"] = rel_lig
                minimized_struct = rel_lig
             except Exception as min_err:
                 logger.error(f"Minimization failed: {min_err}")
                 results["minimized"] = False
        else:
            results["minimized"] = False

        # 6. Aggregation & Output
        # Primary output file: Vina (or Gnina if better/available?)
        # Usually checking mainly Vina for compatibility.
        
        # Calculate Consensus Confidence
        # Simple weighted logic for now or specific TriScore if imported
        results["analysis_depth"] = "Full" if tier_info.get('gnina_enabled') else "Partial"

        # Append Remarks to Output PDBQT
        vina_out = results["engines"].get("vina", {}).get("output_file")
        if vina_out and os.path.exists(vina_out):
            try:
                with open(vina_out, "r+") as f:
                    content = f.read()
                    f.seek(0, 0)
                    f.write("REMARK 200 ========================================================\n")
                    f.write(f"REMARK 200 SMART FUNNEL RESULTS\n")
                    f.write(f"REMARK 200 Tier: {tier_info['tier']} ({tier_info['label']})\n")
                    f.write(f"REMARK 200 Vina (Physics):     {vina_aff:.2f} kcal/mol\n")
                    f.write(f"REMARK 200 Random Forest (ML): {rf_pkd:.2f} pKd\n")
                    if tier_info.get('gnina_enabled'):
                         f.write(f"REMARK 200 Gnina (CNN):        {gnina_aff:.2f} kcal/mol\n")
                    else:
                         f.write(f"REMARK 200 Gnina (CNN):        SKIPPED (Tier Filter)\n")
                    f.write("REMARK 200 ========================================================\n")
                    f.write(content) 
            except Exception as e:
                logger.error(f"Failed to append remarks: {e}")

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

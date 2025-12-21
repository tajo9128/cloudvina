import os
import sys
import subprocess
import logging
from typing import Dict, Optional

# Import helper services
# These must be copied to /app/services in Docker
try:
    # No rDock helper imports needed
    pass
except ImportError:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DockingEngine")

class DockingEngine:
    """
    Unified Docking Interface (Standalone - No ODDT).
    Directly invokes Vina and rDock binaries.
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
        Run Consensus Docking (Vina + rDock + Gnina).
        Returns aggregated results.
        """
        logger.info("Starting CONSENSUS Docking Mode...")
        results = {
            "consensus": True,
            "engines": {},
            "best_affinity": 0.0 # Will be Vina's or Average?
        }
        
        # We need distinct output filenames based on the main output_path (which is usually output.pdbqt or .sdf)
        base_dir = os.path.dirname(output)
        base_name = os.path.splitext(os.path.basename(output))[0]
        
        # 1. Run Vina
        try:
            out_vina = os.path.join(base_dir, f"{base_name}_vina.pdbqt")
            res_vina = self._run_vina(receptor, ligand, out_vina, params)
            results["engines"]["vina"] = res_vina
        except Exception as e:
            logger.error(f"Consensus: Vina failed: {e}")
            results["engines"]["vina"] = {"error": str(e)}


        # 2. Run Gnina (rDock removed - not installed)
        try:
            out_gnina = os.path.join(base_dir, f"{base_name}_gnina.pdbqt")
            res_gnina = self._run_gnina(receptor, ligand, out_gnina, params)
            results["engines"]["gnina"] = res_gnina
        except Exception as e:
            logger.error(f"Consensus: Gnina failed: {e}")
            results["engines"]["gnina"] = {"error": str(e)}


        # Aggregation Logic - Vina + Gnina only
        vina_aff = results["engines"].get("vina", {}).get("best_affinity", 0.0)
        gnina_aff = results["engines"].get("gnina", {}).get("best_affinity", 0.0)
        
        valid_affs = []
        if vina_aff < 0: valid_affs.append(vina_aff)
        if gnina_aff < 0: valid_affs.append(gnina_aff)
        
        if valid_affs:
            results["average_affinity"] = sum(valid_affs) / len(valid_affs)
            results["best_affinity"] = min(valid_affs) # Most negative is best
        
        # Primary output file: Vina (Standard PDBQT)
        # Append Gnina scores as REMARKs to the Vina output file
        vina_out = results["engines"].get("vina", {}).get("output_file")
        
        if vina_out and os.path.exists(vina_out):
            try:
                with open(vina_out, "r+") as f:
                    content = f.read()
                    f.seek(0, 0)
                    f.write("REMARK 200 ========================================================\n")
                    f.write(f"REMARK 200 CONSENSUS DOCKING RESULTS\n")
                    f.write(f"REMARK 200 AutoDock Vina Best: {vina_aff:.2f} kcal/mol\n")
                    f.write(f"REMARK 200 Gnina (AI) Best:    {gnina_aff:.2f} kcal/mol\n")
                    f.write(f"REMARK 200 Average:            {results.get('average_affinity', 0):.2f} kcal/mol\n")
                    f.write("REMARK 200 ========================================================\n")
                    f.write(content) 
                    # Mixing formats is dangerous for parsers, so we stick to remarks for now.
                    # Ideally, we would convert rdock.sdf to PDBQT and append as new MODELs.
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
            # Remove PDB headers that Vina can't parse
            subprocess.run(f"grep -v '^HEADER\|^REMARK\|^AUTHOR\|^REVDAT\|^JRNL' '{receptor}' > '{clean_pdb}'", shell=True, check=True)
            # Convert to PDBQT
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
        
        # Parse output and include execution details
        parsed = self._parse_vina_like_output(output, "vina")
        parsed['stdout'] = result.stdout
        parsed['stderr'] = result.stderr
        parsed['command'] = ' '.join(cmd)
        
        # Parse poses from stdout using VinaParser
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
        # Convert receptor PDB to PDBQT if needed
        if receptor.endswith('.pdb'):
            receptor_pdbqt = receptor.replace('.pdb', '.pdbqt')
            clean_pdb = receptor.replace('.pdb', '_clean.pdb')
            # Remove PDB headers
            subprocess.run(f"grep -v '^HEADER\|^REMARK\|^AUTHOR\|^REVDAT\|^JRNL' '{receptor}' > '{clean_pdb}'", shell=True, check=True)
            # Convert to PDBQT
            subprocess.run(['obabel', clean_pdb, '-O', receptor_pdbqt, '-xr'], check=True)
            receptor = receptor_pdbqt
        
        # Gnina arguments are Vina-compatible
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
            '--exhaustiveness', '4',  # Reduced from 8 to speed up
            '--num_modes', '3'  # Limit output modes for faster completion
        ]
        
        logger.info(f"Running Gnina: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
        except subprocess.TimeoutExpired:
            logger.error("Gnina timed out after 300 seconds")
            raise Exception("Gnina execution timed out after 5 minutes")
        
        if result.returncode != 0:
            # Gnina might output useful info even on failure, but for now specific check
            raise Exception(f"Gnina failed: {result.stderr}")
            
        # Gnina output is PDBQT format, compatible with Vina parser
        # Parse output and include execution details
        parsed = self._parse_vina_like_output(output, "gnina")
        parsed['stdout'] = result.stdout
        parsed['stderr'] = result.stderr
        parsed['command'] = ' '.join(cmd)
        
        # Parse poses from stdout using VinaParser logic (Gnina output is compatible)
        try:
            from services.vina_parser import parse_vina_log
            log_parsed = parse_vina_log(result.stdout)
            if log_parsed and 'poses' in log_parsed:
                parsed['poses'] = log_parsed['poses']
                # Gnina specific: try to extract CNN scores for each pose if present in stdout
                # Standard VinaParser doesn't capture extra columns, but at least we get affinities
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
                    # Vina uses: REMARK VINA RESULT:    -8.5      0.000      0.000
                    if engine_name == 'vina' and line.startswith('REMARK VINA RESULT'):
                        parts = line.split()
                        if len(parts) >= 4:
                            best_affinity = float(parts[3])
                            break
                    
                    # Gnina uses: REMARK minimizedAffinity -8.12345
                    if engine_name == 'gnina' and 'minimizedAffinity' in line:
                        parts = line.split()
                        if len(parts) >= 3:
                            best_affinity = float(parts[2])
                            # Don't break - continue to find CNN score
                    
                    # Gnina CNN score: REMARK CNNscore 0.654321
                    if engine_name == 'gnina' and 'CNNscore' in line:
                        parts = line.split()
                        if len(parts) >= 3:
                            best_cnn_score = float(parts[2])
                            # If we have both, we can break
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
        """Run rDock (rbdock)"""
        job_id = params.get('job_id', 'unknown')
        
        # 1. Convert inputs (PDBQT -> MOL2/SDF)
        try:
            rec_mol2, lig_sdf = prepare_for_rdock(receptor_pdbqt, ligand_pdbqt)
            logger.info(f"Converted inputs: {rec_mol2}, {lig_sdf}")
        except Exception as e:
            raise Exception(f"File conversion failed: {e}")
            
        # 2. Generate Parameter File (.prm)
        prm_file = generate_rdock_config(job_id, rec_mol2, lig_sdf, params)
        
        # 3. Cavity Definition (rbcavity)
        # We need to map the cavity using the 'MOL' method (using ligand as reference)
        # Note: rdock_config generates a prm that defines the mapper.
        # We must run rbcavity -r <prm> -was <output_as>
        cavity_as = f"/tmp/{job_id}.as"
        
        cav_cmd = [self.rbcavity_bin, '-r', prm_file, '-was', cavity_as]
        logger.info(f"Running rbcavity: {' '.join(cav_cmd)}")
        cav_res = subprocess.run(cav_cmd, capture_output=True, text=True)
        
        # rDock logic: rbcavity setup is crucial. If it fails, docking fails.
        # But for 'reference ligand' method, strict cavity is sometimes optional if 
        # using the simple scoring function, but usually required.
        
        # 4. Docking (rbdock)
        # rbdock -i <input_ligand> -o <output_prefix> -r <prm_file> -p <dock_prm> -n <n_runs>
        # Note: rDock output is usually <prefix>.sd - we must handle this.
        out_prefix = os.path.splitext(output_sdf)[0]
        
        # Minimal dock prm:
        dock_prm = "/tmp/dock.prm"
        with open(dock_prm, 'w') as f:
            f.write("RBT_DOCKING_PROTOCOL_V1.00\\n") 
        
        dock_cmd = [
            self.rdock_bin,
            '-i', lig_sdf,
            '-o', out_prefix,
            '-r', prm_file,
            '-p', dock_prm,
            '-n', '10' # 10 runs
        ]
        
        logger.info(f"Running rbdock: {' '.join(dock_cmd)}")
        dock_res = subprocess.run(dock_cmd, capture_output=True, text=True)
        
        # Check output
        # rDock adds .sd extension automatically if not present in prefix
        real_output = f"{out_prefix}.sd"
        if not os.path.exists(real_output):
             # Try output.sdf if it used that
             if os.path.exists(output_sdf):
                 real_output = output_sdf
             else:
                 raise Exception(f"rDock failed: {dock_res.stderr} \nOutput not found.")
                 
        # 5. Parse Results
        results = parse_rdock_output(real_output)
        
        return {
            "best_affinity": results.get('best_affinity', 0.0),
            "output_file": real_output,
            "engine": "rdock"
        }

def run_docking_job(engine, receptor, ligand, output, config=None):
    engine = DockingEngine(engine)
    return engine.run_docking(receptor, ligand, output, config)

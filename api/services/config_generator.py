import os
import boto3
from botocore.exceptions import ClientError
import logging

logger = logging.getLogger("config_generator")

s3_client = boto3.client('s3')
S3_BUCKET = os.getenv('S3_BUCKET', 'BioDockify-jobs-use1-1763775915')

def calculate_autobox_params(receptor_content: str):
    """
    Rigorous Autoboxing following "International-Level" docking protocols.
    
    RULE 1: No-Zero Policy
    - Golden Path: Use HETATM centroid (co-crystallized ligand)
    - Fallback: Use protein geometric center (with warning)
    - NEVER return 0,0,0 without explicit user override
    
    RULE 2: Goldilocks Sizing
    - Standard: 22x22x22 Å (increased from 20 for better coverage)
    - Constraints: 18Å min, 40Å max
    """
    try:
        lines = receptor_content.splitlines()
        
        # Parse coordinates
        het_coords = []
        prot_coords = []
        
        # Expanded list of non-ligand HETATM residues
        # These are ions, cofactors, and crystallization artifacts
        ignore_res = [
            'HOH', 'WAT', 'TIP', 'SOL',  # Water
            'NA', 'CL', 'K', 'BR', 'I',   # Monatomic ions
            'MG', 'ZN', 'CA', 'FE', 'MN', 'CO', 'NI', 'CU',  # Metal ions
            'SO4', 'PO4', 'NO3', 'ACT', 'EDO', 'PEG', 'GOL'  # Common artifacts
        ]
        
        for line in lines:
            if line.startswith("ATOM  "):
                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    prot_coords.append((x, y, z))
                except: 
                    pass
            elif line.startswith("HETATM"):
                res_name = line[17:20].strip()
                if res_name in ignore_res: 
                    continue
                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    het_coords.append((x, y, z))
                except: 
                    pass

        # GOLDEN PATH: Ligand-Based Centering
        target_coords = []
        source = None
        
        # Threshold: At least 8 heavy atoms to qualify as "real ligand"
        # (Filters out residual ions/small molecules)
        if len(het_coords) >= 8:
            target_coords = het_coords
            source = "HETATM (Co-crystallized Ligand)"
            logger.info(f"✓ Golden Path: Found {len(het_coords)} HETATM atoms")
            
        # FALLBACK PATH: Protein Geometric Center (Blind Docking)
        elif prot_coords:
            target_coords = prot_coords
            source = "Protein Center (Blind Mode)"
            logger.warning(f"⚠️ No co-crystallized ligand detected. Using protein center (accuracy may be reduced).")
            
        else:
            # CRITICAL FAILURE: Empty structure
            logger.error("❌ Autoboxing FAILED: No parseable atoms in receptor.")
            return None

        # Calculate Centroid
        min_x = min(c[0] for c in target_coords)
        max_x = max(c[0] for c in target_coords)
        min_y = min(c[1] for c in target_coords)
        max_y = max(c[1] for c in target_coords)
        min_z = min(c[2] for c in target_coords)
        max_z = max(c[2] for c in target_coords)
        
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        center_z = (min_z + max_z) / 2.0
        
        # GOLDILOCKS SIZING
        # If ligand-based: Tight box (ligand extent + 10Å buffer)
        # If protein-based: Larger box (but capped at 40Å for accuracy)
        
        if source.startswith("HETATM"):
            # Ligand-based: Snug fit
            padding = 10.0
            size_x = (max_x - min_x) + padding
            size_y = (max_y - min_y) + padding
            size_z = (max_z - min_z) + padding
        else:
            # Blind docking: Cover ~60% of protein, capped
            span_x = max_x - min_x
            span_y = max_y - min_y
            span_z = max_z - min_z
            size_x = min(span_x * 0.6, 40.0)
            size_y = min(span_y * 0.6, 40.0)
            size_z = min(span_z * 0.6, 40.0)
        
        # Enforce International Standard: 22Å default, 18Å min, 40Å max
        size_x = max(min(size_x, 40.0), 18.0)
        size_y = max(min(size_y, 40.0), 18.0)
        size_z = max(min(size_z, 40.0), 18.0)
        
        # Default to 22x22x22 if calculated size is close to minimum
        if size_x < 20.0: size_x = 22.0
        if size_y < 20.0: size_y = 22.0
        if size_z < 20.0: size_z = 22.0

        logger.info(f"📦 Autobox: {source} → Center({center_x:.2f}, {center_y:.2f}, {center_z:.2f}) Size({size_x:.1f}Å × {size_y:.1f}Å × {size_z:.1f}Å)")
        
        return {
            'grid_center_x': round(center_x, 3), 
            'grid_center_y': round(center_y, 3), 
            'grid_center_z': round(center_z, 3),
            'grid_size_x': round(size_x, 1), 
            'grid_size_y': round(size_y, 1), 
            'grid_size_z': round(size_z, 1),
            'autobox_source': source
        }

    except Exception as e:
        logger.error(f"❌ Autoboxing exception: {e}")
        return None

def generate_vina_config(job_id: str, grid_params: dict = None, receptor_content: str = None):
    """
    Generate AutoDock Vina config file and store in S3
    
    Args:
        job_id: Job ID
        grid_params: Dict with grid_center_x/y/z and grid_size_x/y/z.
        receptor_content: Optional PDB/PDBQT content for Autoboxing fallback.
    
    Returns:
        S3 key where config was stored
    """
    # Default grid parameters (safe fallback)
    params = grid_params or {}
    
    # Check if params are "Empty" (all zeros), which implies user wants Autoboxing
    is_zero_center = (params.get('grid_center_x', 0) == 0 and 
                      params.get('grid_center_y', 0) == 0 and 
                      params.get('grid_center_z', 0) == 0)

    # Attempt Autoboxing if requested or missing params
    autobox_source = "Manual"
    if (not grid_params or is_zero_center) and receptor_content:
        auto_params = calculate_autobox_params(receptor_content)
        if auto_params:
            params.update(auto_params)
            autobox_source = "Auto-Computed"

    center_x = params.get('grid_center_x', 0.0)
    center_y = params.get('grid_center_y', 0.0)
    center_z = params.get('grid_center_z', 0.0)
    size_x = params.get('grid_size_x', 20)
    size_y = params.get('grid_size_y', 20)
    size_z = params.get('grid_size_z', 20)
    exhaustiveness = params.get('exhaustiveness', 16) # Balanced (User Request)
    num_modes = params.get('num_modes', 5) # Focused (User Request)
    energy_range = params.get('energy_range', 5)
    
    # Generate config content
    config_content = f"""# AutoDock Vina Configuration File
# Generated by BioDockify ({autobox_source})
# Job ID: {job_id}

# Input files
receptor = receptor.pdbqt
ligand = ligand.pdbqt

# Grid box configuration
center_x = {center_x}
center_y = {center_y}
center_z = {center_z}

size_x = {size_x}
size_y = {size_y}
size_z = {size_z}

# Docking parameters
exhaustiveness = {exhaustiveness}
num_modes = {num_modes}
energy_range = {energy_range}

# Output
out = output.pdbqt
"""
    
    # Store in S3
    config_key = f"jobs/{job_id}/config.txt"
    
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=config_key,
            Body=config_content.encode('utf-8'),
            ContentType='text/plain'
        )
        
        return config_key
    
    except ClientError as e:
        print(f"Error storing config file: {e}")
        raise


def get_presigned_download_url(s3_key: str, expiration=3600):
    """
    Generate presigned URL for downloading from S3
    """
    try:
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': S3_BUCKET, 'Key': s3_key},
            ExpiresIn=expiration
        )
        return url
    except ClientError as e:
        print(f"Error generating presigned URL: {e}")
        return None

import os
import boto3
from botocore.exceptions import ClientError

s3_client = boto3.client('s3')
S3_BUCKET = os.getenv('S3_BUCKET', 'BioDockify-jobs-use1-1763775915')

def generate_rdock_config(job_id: str, receptor_file: str, ligand_file: str, grid_params: dict = None) -> str:
    """
    Generate rDock parameter file (.prm).
    
    Args:
        job_id: Unique Job ID
        receptor_file: Local path to receptor file (.mol2)
        ligand_file: Local path to ligand file (.sd/.mol2)
        grid_params: Dict with center/size (from CavityDetector)
    
    Returns:
        Path to local .prm file
    """
    params = grid_params or {}
    
    # rDock uses a cavity definition (Mapper).
    # Simple approach: Define a sphere around the cavity center.
    center = [
        params.get('center_x', 0.0),
        params.get('center_y', 0.0),
        params.get('center_z', 0.0)
    ]
    radius = max(
        params.get('size_x', 20.0),
        params.get('size_y', 20.0),
        params.get('size_z', 20.0)
    ) / 2.0  # Convert box size to rough radius
    
    # RBT_PARAMETER_FILE_V1.00
    prm_content = f"""RBT_PARAMETER_FILE_V1.00
TITLE BioDockify_Job_{job_id}

RECEPTOR_FILE {receptor_file}
RECEPTOR_FLEX 3.0

##################################################################
# CAVITY DEFINITION: REFERENCE LIGAND METHOD
# We don't have a ref ligand, so we use SPHERE method if rbcavity supports it
# Or we rely on the user providing a reference.
# For generic blind docking or box-based docking, rDock is trickier than Vina.
#
# STRATEGY: 
# 1. Use the "MOL" mapper method if we have a cavity center.
#    This creates a pseudo-ligand at the center to define the cavity.
##################################################################

SECTION MAPPER
    SITE_MAPPER RbtLigandElementMap # Default mapper
    REF_MOL {ligand_file}  # Use input ligand as start reference (if aligned) or..
    radius {radius}
    small_sphere 1.0
    min_volume 100
    max_cavities 1
    vol_incr 0.0
    gridstep 0.5
END_SECTION

SECTION CAVITY
    SCORING_FUNCTION RbtCavityGridSF
    WEIGHT 1.0
END_SECTION

SECTION PHARMA
    SCORING_FUNCTION RbtPharmaSF
    WEIGHT 1.0
    CONSTRAINTS FALSE
END_SECTION

SECTION SCORE
    SCORING_FUNCTION RbtLigandSF
    WEIGHT 1.0
END_SECTION

"""
    # Note: The above is a simplification. rDock cavity definition is complex.
    # A more robust "Box" to "Cavity" mapping is usually done by creating a dummy .sd file
    # at the center coordinates and running rbcavity on it.
    
    prm_filename = f"/tmp/{job_id}.prm"
    with open(prm_filename, 'w') as f:
        f.write(prm_content)
        
    return prm_filename

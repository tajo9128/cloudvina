import subprocess
import os

def convert_format(input_path: str, output_format: str) -> str:
    """
    Convert chemical file format using OpenBabel (obabel).
    
    Args:
        input_path: Path to input file (e.g., 'receptor.pdbqt')
        output_format: Target extension (e.g., 'mol2', 'sdf')
        
    Returns:
        Path to converted file
    """
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}.{output_format}"
    
    # Map 'sdf' to 'sdf' (obabel uses 'sdf' or 'sd')
    obabel_format = output_format
    if output_format == 'sdf':
        obabel_format = 'sd'
        
    cmd = [
        "obabel",
        f"-i{ext[1:]}", # input format (remove dot)
        input_path,
        f"-o{obabel_format}", 
        f"-O{output_path}",
        "--gen3d" # Generate 3D coordinates if missing (critical for SMILES/2D)
    ]
    
    # For PDBQT -> MOL2 receptor conversion, we might need to preserve charges or hydrogens
    if ext == '.pdbqt' and output_format == 'mol2':
        cmd.append("-p") # Add hydrogens appropriate for pH 7.4
        
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed: {e.stderr.decode()}")
        raise Exception(f"File conversion failed: {e.stderr.decode()}")

def prepare_for_rdock(receptor_pdbqt: str, ligand_pdbqt: str):
    """
    Convert Vina inputs (PDBQT) to rDock inputs (MOL2/SDF).
    """
    # Receptor: PDBQT -> MOL2
    receptor_mol2 = convert_format(receptor_pdbqt, 'mol2')
    
    # Ligand: PDBQT -> SD (SDF is best for rDock output handling)
    # Note: rDock prefers .sd for ligands
    ligand_sdf = convert_format(ligand_pdbqt, 'sdf')
    
    return receptor_mol2, ligand_sdf

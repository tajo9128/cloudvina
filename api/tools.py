from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import Response
import io
from rdkit import Chem
from meeko import MoleculePreparation
from meeko import PDBQTMolecule

router = APIRouter(prefix="/tools", tags=["tools"])

@router.post("/convert/sdf-to-pdbqt")
async def convert_sdf_to_pdbqt(file: UploadFile = File(...)):
    """
    Convert an uploaded SDF file to PDBQT format.
    Returns the PDBQT content as a file download.
    """
    if not file.filename.lower().endswith('.sdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Please upload an .sdf file."
        )

    try:
        # Read file content
        content = await file.read()
        sdf_string = content.decode('utf-8')

        # Parse SDF with RDKit
        # Try parsing as a single block first
        mol = Chem.MolFromMolBlock(sdf_string, removeHs=False)
        
        if mol is None:
            # If that fails, try using ForwardSDMolSupplier with a memory stream
            # This handles cases with multiple molecules (taking the first) or slightly malformed files
            stream = io.BytesIO(content)
            suppl = Chem.ForwardSDMolSupplier(stream, removeHs=False)
            try:
                mol = next(suppl)
            except StopIteration:
                mol = None
            except Exception:
                mol = None
            
        if mol is None:
            raise ValueError("Could not parse SDF file. Ensure it contains valid 3D molecular data.")

        # Add hydrogens if missing (important for docking)
        mol = Chem.AddHs(mol, addCoords=True)

        # Prepare PDBQT with Meeko
        preparator = MoleculePreparation()
        preparator.prepare(mol)
        pdbqt_string = preparator.write_pdbqt_string()

        # Return as file download
        return Response(
            content=pdbqt_string,
            media_type="chemical/x-pdbqt",
            headers={
                "Content-Disposition": f"attachment; filename={file.filename.replace('.sdf', '.pdbqt')}"
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Conversion failed: {str(e)}"
        )

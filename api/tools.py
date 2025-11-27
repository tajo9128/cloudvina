from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import Response
import io
from rdkit import Chem
from meeko import MoleculePreparation

router = APIRouter(prefix="/tools", tags=["Tools"])

@router.post("/convert-to-pdbqt")
async def convert_to_pdbqt(file: UploadFile = File(...)):
    """
    Convert various molecule formats (SDF, PDB, MOL, MOL2) to PDBQT
    """
    try:
        content = await file.read()
        filename = file.filename.lower()
        
        # Determine format and parse
        mol = None
        if filename.endswith('.sdf'):
            suppl = Chem.SDMolSupplier()
            suppl.SetData(content)
            mol = next(suppl)
        elif filename.endswith('.pdb'):
            mol = Chem.MolFromPDBBlock(content.decode('utf-8'))
        elif filename.endswith('.mol'):
            mol = Chem.MolFromMolBlock(content.decode('utf-8'))
        elif filename.endswith('.mol2'):
            mol = Chem.MolFromMol2Block(content.decode('utf-8'))
        else:
            # Try generic parsing if extension doesn't match
            mol = Chem.MolFromMolBlock(content.decode('utf-8'))
            
        if not mol:
            raise ValueError("Could not parse molecule file")

        # Prepare PDBQT
        mol = Chem.AddHs(mol, addCoords=True)
        preparator = MoleculePreparation()
        preparator.prepare(mol)
        pdbqt_string = preparator.write_pdbqt_string()

        return Response(
            content=pdbqt_string,
            media_type="chemical/x-pdbqt",
            headers={
                "Content-Disposition": f"attachment; filename={file.filename.rsplit('.', 1)[0]}.pdbqt"
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Conversion failed: {str(e)}"
        )

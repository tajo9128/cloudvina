from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import Response
import io
from rdkit import Chem
from meeko import MoleculePreparation

router = APIRouter(prefix="/tools", tags=["Tools"])

import gzip

@router.post("/convert-to-pdbqt")
async def convert_to_pdbqt(file: UploadFile = File(...)):
    """
    Convert various molecule formats (SDF, PDB, MOL, MOL2) to PDBQT
    """
    try:
        content = await file.read()
        filename = file.filename.lower()

        # Handle GZIP compression
        if content.startswith(b'\x1f\x8b'):
            try:
                content = gzip.decompress(content)
                if filename.endswith('.gz'):
                    filename = filename[:-3]
            except Exception as e:
                # If it looks like gzip but fails, it's likely corrupt.
                # Raise error instead of falling through to decode error.
                raise ValueError(f"File appears to be GZIP compressed but failed to decompress: {str(e)}")
        
        # Determine format and parse
        mol = None
        if filename.endswith('.sdf'):
            # Use ForwardSDMolSupplier for in-memory bytes
            suppl = Chem.ForwardSDMolSupplier(io.BytesIO(content))
            try:
                mol = next(suppl)
            except StopIteration:
                raise ValueError("Empty SDF file")
        elif filename.endswith('.pdb'):
            mol = Chem.MolFromPDBBlock(content.decode('utf-8'))
        elif filename.endswith('.mol'):
            mol = Chem.MolFromMolBlock(content.decode('utf-8'))
        elif filename.endswith('.mol2'):
            mol = Chem.MolFromMol2Block(content.decode('utf-8'))
        else:
            # Try generic parsing if extension doesn't match
            try:
                decoded = content.decode('utf-8')
                mol = Chem.MolFromMolBlock(decoded)
            except UnicodeDecodeError:
                 raise ValueError("File is not a valid text molecule file and not a recognized binary format")
            
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

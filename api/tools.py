from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import Response
import io
import gzip
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation

router = APIRouter(prefix="/tools", tags=["Tools"])

def parse_cif_to_mol(content: bytes) -> Chem.Mol:
    """Parse CIF/mmCIF format using BioPython and convert to RDKit Mol"""
    try:
        from Bio.PDB import MMCIFParser
        from io import StringIO
        
        # Parse CIF structure
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("protein", StringIO(content.decode('utf-8')))
        
        # Convert to PDB format string (RDKit can read this)
        from Bio.PDB import PDBIO
        pdb_io = PDBIO()
        pdb_io.set_structure(structure)
        
        pdb_string_io = StringIO()
        pdb_io.save(pdb_string_io)
        pdb_string = pdb_string_io.getvalue()
        
        # Parse with RDKit
        mol = Chem.MolFromPDBBlock(pdb_string)
        return mol
    except Exception as e:
        raise ValueError(f"CIF parsing failed: {str(e)}")

def parse_pdbml_xml(content: bytes) -> Chem.Mol:
    """Parse PDBML XML format using BioPython"""
    try:
        from Bio.PDB import PDBMLParser
        from io import StringIO
        
        # Parse XML structure
        parser = PDBMLParser()
        structure = parser.get_structure("protein", StringIO(content.decode('utf-8')))
        
        # Convert to PDB format string
        from Bio.PDB import PDBIO
        pdb_io = PDBIO()
        pdb_io.set_structure(structure)
        
        pdb_string_io = StringIO()
        pdb_io.save(pdb_string_io)
        pdb_string = pdb_string_io.getvalue()
        
        # Parse with RDKit
        mol = Chem.MolFromPDBBlock(pdb_string)
        return mol
    except Exception as e:
        raise ValueError(f"XML/PDBML parsing failed: {str(e)}")

@router.post("/convert-to-pdbqt")
async def convert_to_pdbqt(file: UploadFile = File(...)):
    """
    Convert various molecule formats to PDBQT for AutoDock Vina
    Supported formats: PDB, PDBQT, SDF, MOL, MOL2, CIF, mmCIF, PQR, XML/PDBML, SMILES (.gz supported)
    """
    try:
        content = await file.read()
        filename = file.filename.lower()

        # Handle GZIP compression
        if content.startswith(b'\\x1f\\x8b'):
            try:
                content = gzip.decompress(content)
                if filename.endswith('.gz'):
                    filename = filename[:-3]
            except Exception as e:
                raise ValueError(f"File appears to be GZIP compressed but failed to decompress: {str(e)}")
        
        # Determine format and parse
        mol = None
        
        # Standard formats
        if filename.endswith('.sdf'):
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
        
        # New formats
        elif filename.endswith(('.cif', '.mmcif')):
            mol = parse_cif_to_mol(content)
            
        elif filename.endswith('.pqr'):
            # PQR is PDB with extra columns (charges, radii) - treat as PDB
            mol = Chem.MolFromPDBBlock(content.decode('utf-8'))
            
        elif filename.endswith('.xml') or filename.endswith('.pdbml'):
            mol = parse_pdbml_xml(content)
            
        elif filename.endswith(('.smi', '.smiles')):
            # SMILES format - need to generate 3D coordinates
            smiles_str = content.decode('utf-8').strip().split()[0]  # First column is SMILES
            mol = Chem.MolFromSmiles(smiles_str)
            if mol:
                # Generate 3D coordinates
                mol = Chem.AddHs(mol)
                result = AllChem.EmbedMolecule(mol, randomSeed=42)
                if result != 0:
                    raise ValueError("Failed to generate 3D coordinates from SMILES. Try uploading a 3D structure instead.")
                AllChem.MMFFOptimizeMolecule(mol)  # Optimize geometry
        
        else:
            # Try generic parsing if extension doesn't match
            try:
                decoded = content.decode('utf-8')
                mol = Chem.MolFromMolBlock(decoded)
            except UnicodeDecodeError:
                raise ValueError("File is not a valid text molecule file and not a recognized binary format")
            
        if not mol:
            raise ValueError(f"Could not parse molecule file. Ensure the file is valid and try: PDB, SDF, MOL, MOL2, CIF, PQR, XML, or SMILES format.")

        # Prepare PDBQT (skip AddHs if already done for SMILES)
        if not filename.endswith(('.smi', '.smiles')):
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
@router.post("/fix-cors")
async def fix_s3_cors():
    """
    Forcefully apply CORS configuration to the S3 bucket using backend credentials.
    This resolves browser blocking issues for results visualization.
    """
    try:
        import boto3
        from aws_services import S3_BUCKET, AWS_REGION
        
        print(f"Applying CORS to bucket: {S3_BUCKET} in {AWS_REGION}")
        s3 = boto3.client('s3', region_name=AWS_REGION)
        
        cors_configuration = {
            'CORSRules': [{
                'AllowedHeaders': ['*'],
                'AllowedMethods': ['GET', 'PUT', 'POST', 'DELETE', 'HEAD'],
                'AllowedOrigins': ['*'],
                'ExposeHeaders': ['ETag', 'x-amz-server-side-encryption', 'x-amz-request-id', 'x-amz-id-2'],
                'MaxAgeSeconds': 3000
            }]
        }

        s3.put_bucket_cors(Bucket=S3_BUCKET, CORSConfiguration=cors_configuration)
        return {"status": "success", "message": f"CORS rules applied to {S3_BUCKET}"}
        
    except Exception as e:
        print(f"CORS Fix Failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to apply CORS: {str(e)}")

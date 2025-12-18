
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from io import BytesIO
from rdkit import Chem
from rdkit.Chem import Draw
import os

class ReportGenerator:
    """
    Generates PDF reports for Drug Discovery Pipelines.
    Includes summary tables and 2D structure visualizations.
    """

    def generate_report(self, hits: list, project_name: str = "BioDockify Project") -> BytesIO:
        """
        Generates a PDF report from a list of ranked hits.
        Returns a BytesIO buffer containing the PDF.
        """
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()

        # Title
        title_style = styles['Title']
        elements.append(Paragraph(f"Lead Optimization Report: {project_name}", title_style))
        elements.append(Spacer(1, 12))
        
        # Summary Text
        summary = f"This report contains the top {len(hits)} candidates ranked by consensus scoring (Docking + MM-GBSA + ADMET)."
        elements.append(Paragraph(summary, styles['Normal']))
        elements.append(Spacer(1, 20))

        # Table Data
        # Columns: Rank, ID, Structure, Vina (kcal/mol), Gnina (Score), ADMET
        data = [['Rank', 'ID', 'Structure', 'Vina\n(kcal/mol)', 'Gnina\n(Score)', 'ADMET']]
        
        for hit in hits:
            # 1. Structure Image
            img = self._mol_to_image(hit.get('compound_name', ''), hit.get('smiles', ''))
            
            # 2. Scores - Handle Flat or Nested (Consensus) structure
            # Flat: hit['vina_score'], hit['gnina_score']
            # Nested: hit['engines']['vina']['best_affinity']
            
            vina_score = hit.get('vina_score')
            if vina_score is None and 'engines' in hit and 'vina' in hit['engines']:
                 vina_score = hit['engines']['vina'].get('best_affinity')
            
            gnina_score = hit.get('gnina_score')
            if gnina_score is None and 'engines' in hit and 'gnina' in hit['engines']:
                 gnina_score = hit['engines']['gnina'].get('best_affinity')
            
            # Format
            vina_text = f"{vina_score:.2f}" if isinstance(vina_score, (int, float)) else (str(vina_score) if vina_score else "N/A")
            gnina_text = f"{gnina_score:.2f}" if isinstance(gnina_score, (int, float)) else (str(gnina_score) if gnina_score else "N/A")

            if 'consensus_score' in hit:
                # Add tiny note?
                pass

            # 3. ADMET Verdict
            admet = hit.get('admet', {})
            verdict = "N/A"
            if admet:
                 verdict = f"{admet.get('verdict', 'Unknown')}\n"
                 if admet.get('bbb', {}).get('permeable'):
                     verdict += "(Brain)"

            row = [
                str(hit.get('rank', '-')),
                hit.get('id', 'Unknown')[:8], 
                img if img else "No Structure",
                vina_text,
                gnina_text,
                verdict
            ]
            data.append(row)

        # Create Table
        # Widths: Rank, ID, Struct, Vina, Gnina, ADMET
        col_widths = [0.5*inch, 0.8*inch, 2*inch, 1*inch, 1*inch, 1.5*inch]
        t = Table(data, colWidths=col_widths)
        
        # Table Style
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        elements.append(t)
        doc.build(elements)
        
        buffer.seek(0)
        return buffer

    def generate_pymol_script(self, job_id, receptor_url, ligand_url) -> BytesIO:
        """
        Generates a .pml script to visualize results in PyMOL.
        """
        script = f"""
        # BioDockify PyMOL Script for Job {job_id}
        # Generated automatically

        # 1. Load Files
        load {receptor_url}, receptor
        load {ligand_url}, ligand

        # 2. visual settings
        bg_color white
        show cartoon, receptor
        color gray80, receptor
        show licorice, ligand
        util.cba(154, "ligand")

        # 3. Center view
        zoom ligand
        
        # 4. Show interactions (simple polar guess)
        dist polar_contacts, receptor, ligand, mode=2
        """
        
        buffer = BytesIO()
        buffer.write(script.encode('utf-8'))
        buffer.seek(0)
        return buffer

    def _mol_to_image(self, name: str, smiles: str):
        """Convert SMILES to a ReportLab Image object"""
        try:
            # Check for empty/None inputs
            if not smiles or not isinstance(smiles, str):
                return None
            
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return None
            
            # Generate 2D coords if not present
            if not mol.GetNumConformers():
                from rdkit.Chem import AllChem
                AllChem.Compute2DCoords(mol)
                
            img_data = Draw.MolToImage(mol, size=(150, 150))
            
            # Save to bytes for ReportLab
            img_io = BytesIO()
            img_data.save(img_io, format='PNG')
            img_io.seek(0)
            
            return Image(img_io, width=1.5*inch, height=1.5*inch)
            
        except Exception as e:
            # Fallback for any imaging error (fonts, cairo, libs, etc.)
            print(f"Warning: Could not generate image for {name} ({smiles}): {e}")
            return None

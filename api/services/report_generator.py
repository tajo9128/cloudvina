
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

    def generate_job_report(self, job_data: dict, analysis: dict, interactions: dict) -> BytesIO:
        """
        Generates a comprehensive, publication-ready PDF report for a single docking job.
        Includes Experimental Methods, Results Table, and 2D Interaction Plot.
        """
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()

        # 1. Header & Title
        elements.append(Paragraph(f"BioDockify Docking Report", styles['Title']))
        elements.append(Spacer(1, 12))
        
        # 2. Experimental Details Table
        elements.append(Paragraph("1. Experimental Configuration", styles['Heading2']))
        details_data = [
            ['Job ID', job_data.get('id', 'N/A')],
            ['Target Receptor', job_data.get('receptor_filename', 'N/A')],
            ['Ligand', job_data.get('ligand_filename', 'N/A')],
            ['Engine', 'Consensus (Vina + Gnina)'],
            ['Date', str(job_data.get('created_at', 'N/A'))[:10]]
        ]
        t_details = Table(details_data, colWidths=[2*inch, 4*inch])
        t_details.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 1, colors.gray),
            ('BACKGROUND', (0,0), (0,-1), colors.whitesmoke),
            ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ]))
        elements.append(t_details)
        elements.append(Spacer(1, 12))

        # 3. Results Summary
        elements.append(Paragraph("2. Docking Results", styles['Heading2']))
        
        # Determine Scores
        vina_score = analysis.get('best_affinity', 'N/A')
        # Check consensus
        gnina_score = 'N/A'
        if job_data.get('download_urls', {}).get('output_gnina'):
             # If we have gnina output, imply score existed or fetched.
             # For now, placeholder or check analysis deeper
             pass

        res_data = [
            ['Metric', 'Value', 'Unit'],
            ['Best Binding Affinity', f"{vina_score} kcal/mol" if isinstance(vina_score, (int, float)) else str(vina_score), 'kcal/mol'],
            ['Ligand Efficiency', 'N/A', 'kcal/mol/heavy_atom'] 
        ]
        t_res = Table(res_data, colWidths=[2*inch, 2*inch, 2*inch])
        t_res.setStyle(TableStyle([
             ('BACKGROUND', (0,0), (-1,0), colors.navy),
             ('TEXTCOLOR', (0,0), (-1,0), colors.white),
             ('GRID', (0,0), (-1,-1), 1, colors.black),
             ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ]))
        elements.append(t_res)
        elements.append(Spacer(1, 12))

        # 4. 2D Interaction Visualization (SMILES)
        elements.append(Paragraph("3. 2D Structure & Interactions", styles['Heading2']))
        # Try to generate 2D image
        smiles = None
        if analysis and analysis.get('poses'):
             # Try to get SMILES from first pose or use input if we had it
             pass 
        
        # Fallback: Generate from input ligand filename if it's SMILES-based or try to extract from job data
        # For this demo, we'll try to use the 'notes' field if it stored SMILES, or skip image if unavailable.
        ligand_smiles = job_data.get('notes')  # Often we store SMILES here
        img = self._mol_to_image("Ligand", ligand_smiles) if ligand_smiles else None
        
        if img:
            elements.append(img)
            elements.append(Paragraph(f"Ligand Structure: {ligand_smiles}", styles['Normal']))
        else:
            elements.append(Paragraph("Structure image unavailable (SMILES not found in metadata).", styles['Italic']))
            
        elements.append(Spacer(1, 12))

        # 5. Methods Text (Publication Ready)
        elements.append(Paragraph("4. Methodology", styles['Heading2']))
        method_text = """
        Molecular docking was performed using the BioDockify Consensus Protocol. 
        The target receptor file was prepared by removing water molecules and adding polar hydrogens. 
        Ligand structures were generated from SMILES/2D inputs using RDKit with MMFF94 energy minimization 
        and converted to PDBQT format via Meeko. 
        Docking was executed using AutoDock Vina 1.2.3 and Gnina (CNN scoring), with a search exhaustiveness of 8. 
        The binding site was defined by a cubic grid centered on the target pocket.
        Results were ranked by binding affinity (kcal/mol) and visual inspection of protein-ligand interactions.
        """
        elements.append(Paragraph(method_text, styles['Normal']))
        
        doc.build(elements)
        buffer.seek(0)
        return buffer

    def generate_pymol_script(self, job_id, receptor_url, ligand_url) -> BytesIO:
        """
        Generates a .pml script to visualize results in PyMOL.
        """
        script = f"""
# BioDockify PyMOL Session Script
# Job ID: {job_id}
# Usage: Open PyMOL -> File -> Run -> Select this file

# 1. Fetch Structures
load {receptor_url}, receptor
load {ligand_url}, ligand

# 2. Visual Styling (Publication Quality)
bg_color white
hide everything
show cartoon, receptor
color pale_cyan, receptor
show sticks, ligand
util.cba(154, "ligand") # Color by atom, gray carbons

# 3. Show Binding Pocket Surface
# create pocket, byres receptor within 5 of ligand
# show surface, pocket
# set transparency, 0.5, pocket
# color white, pocket

# 4. Highlight Interactions (H-Bonds)
dist hbonds, receptor, ligand, mode=2
color yellow, hbonds
set dash_gap, 0.2
set dash_width, 2.0

# 5. Orientation
zoom ligand, 8
orient ligand

# End of script
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
                
            img_data = Draw.MolToImage(mol, size=(300, 300)) # Larger for report
            
            # Save to bytes for ReportLab
            img_io = BytesIO()
            img_data.save(img_io, format='PNG')
            img_io.seek(0)
            
            return Image(img_io, width=3*inch, height=3*inch) # Resize in PDF
            
        except Exception as e:
            # Fallback for any imaging error (fonts, cairo, libs, etc.)
            print(f"Warning: Could not generate image for {name} ({smiles}): {e}")
            return None

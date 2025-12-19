import io
import os
import uuid
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem

# Supabase client (passed from route to avoid circular imports or re-init)

def get_logo_path():
    # Placeholder for logo, can be added later
    return None

def generate_structure_image(smiles: str) -> io.BytesIO:
    """Generates a 2D structure image from SMILES."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            # Try to handle PDBQT if SMILES not available? 
            # Ideally we have SMILES in the database.
            # If not, return None
            return None
        
        AllChem.Compute2DCoords(mol)
        img = Draw.MolToImage(mol, size=(200, 200))
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        return img_buffer
    except Exception:
        return None

def generate_batch_pdf(batch_id: str, jobs_data: list, batch_meta: dict) -> io.BytesIO:
    """
    Generates a PDF report for a docking batch.
    
    Args:
        batch_id: ID of the batch
        jobs_data: List of dictionaries containing job details (affinity, smiles, etc.)
        batch_meta: Dictionary with batch metadata (created_at, total jobs)
        
    Returns:
        io.BytesIO object containing the PDF
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CenterTitle', parent=styles['Heading1'], alignment=TA_CENTER, spaceAfter=20))
    styles.add(ParagraphStyle(name='SectionHeader', parent=styles['Heading2'], spaceBefore=15, spaceAfter=10))
    styles.add(ParagraphStyle(name='NormalSmall', parent=styles['Normal'], fontSize=9))

    elements = []
    
    # --- Title Page ---
    elements.append(Paragraph("BioDockify SAR Report", styles['CenterTitle']))
    elements.append(Spacer(1, 12))
    
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    elements.append(Paragraph(f"<b>Batch ID:</b> {batch_id}", styles['Normal']))
    elements.append(Paragraph(f"<b>Date Generated:</b> {date_str}", styles['Normal']))
    elements.append(Paragraph(f"<b>Receptor:</b> {jobs_data[0].get('receptor_filename', 'Unknown') if jobs_data else 'N/A'}", styles['Normal']))
    elements.append(Spacer(1, 24))
    
    # --- Executive Summary ---
    elements.append(Paragraph("Executive Summary", styles['SectionHeader']))
    
    # Calculate Stats
    total_jobs = len(jobs_data)
    succeeded = [j for j in jobs_data if j.get('status') == 'SUCCEEDED']
    failed = [j for j in jobs_data if j.get('status') == 'FAILED']
    
    affinities = []
    for j in succeeded:
        try:
            aff = float(j.get('binding_affinity', 0))
            if aff < 0: # Sanity check for Vina scores
                affinities.append(aff)
        except:
            pass
            
    best_affinity = min(affinities) if affinities else "N/A"
    avg_affinity = sum(affinities) / len(affinities) if affinities else "N/A"
    
    # Count "Hits" (< -8.0 typically, or -9.0 for "Premium")
    hits = [a for a in affinities if a <= -9.0]
    hit_rate = (len(hits) / total_jobs * 100) if total_jobs > 0 else 0
    
    summary_data = [
        ["Metric", "Value"],
        ["Total Compounds", str(total_jobs)],
        ["Successful Docks", str(len(succeeded))],
        ["Best Affinity", f"{best_affinity} kcal/mol" if isinstance(best_affinity, float) else best_affinity],
        ["Average Affinity", f"{avg_affinity:.2f} kcal/mol" if isinstance(avg_affinity, float) else avg_affinity],
        ["High Affinity Hits (<-9.0)", f"{len(hits)} ({hit_rate:.1f}%)"]
    ]
    
    t = Table(summary_data, colWidths=[3*inch, 2*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.HexColor("#6366f1")),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor("#e2e8f0")),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 24))
    
    # --- Scatter Plot (Affinity Distribution) ---
    if affinities:
        plt.figure(figsize=(6, 4))
        plt.hist(affinities, bins=15, color='#6366f1', alpha=0.7, rwidth=0.85)
        plt.title('Binding Affinity Distribution')
        plt.xlabel('Affinity (kcal/mol)')
        plt.ylabel('Count')
        plt.grid(axis='y', alpha=0.5)
        
        plot_buffer = io.BytesIO()
        plt.savefig(plot_buffer, format='png', dpi=100)
        plt.close()
        plot_buffer.seek(0)
        
        elements.append(Paragraph("Affinity Distribution", styles['SectionHeader']))
        elements.append(Image(plot_buffer, width=5*inch, height=3.3*inch))
        elements.append(Spacer(1, 24))

    elements.append(PageBreak())

    # --- Top Hits Detailed View ---
    elements.append(Paragraph("Top Candidates (Ranked)", styles['CenterTitle']))
    
    # Sort by affinity
    # Ensure numerical sort
    def get_affinity(j):
        try:
            val = float(j.get('binding_affinity', 0))
            return val if val < 0 else 0
        except:
            return 0
            
    ranked_jobs = sorted(succeeded, key=get_affinity)[:10] # Top 10
    
    # Create rows for each top hit
    for i, job in enumerate(ranked_jobs):
        # Data container
        stats_text = [
            f"<b>Rank:</b> {i+1}",
            f"<b>Name:</b> {job.get('ligand_filename', 'Unknown')}",
            f"<b>Affinity:</b> {job.get('binding_affinity')} kcal/mol",
            f"<b>Job ID:</b> {job.get('id')}"
        ]
        
        # Add ADMET if available (Phase 9 feature check)
        # Note: In a real implementation we'd check job['admet_summary'] or similar
        # For now, just basic binding info
        
        # Generate Image
        # Note: We rely on 'ligand_s3_key' or we'd need SMILES.
        # If we stored SMILES in the job record during CSV upload, use it.
        # Otherwise, skip image for now or fetch it (too costly for PDF generation to fetch S3 for every image).
        # Assuming CSV upload provides 'smiles' in a metadata field if applicable, or we stored it.
        # Let's check if the generic 'smiles' field exists on the job model.
        # If not, we'll skip the image or use a placeholder.
        
        # Ideally, our CSV batch submitter should save SMILES. 
        # But looking at `batch.py`, we didn't explicitly save 'smiles' to a column, only PDBQT to S3.
        # Wait, the CSV submitter `submit_csv_batch` converts SMILES to PDBQT.
        # It doesn't save the SMILES string to the DB `jobs` table (schema review might be needed).
        # However, for now, we'll list the text details.
        
        # Create a mini table for this row
        row_data = [[
            Paragraph("<br/>".join(stats_text), styles['Normal']),
            # Right side: Blank for now unless we have image
            "" 
        ]]
        
        t = Table(row_data, colWidths=[4*inch, 2*inch])
        t.setStyle(TableStyle([
            ('BOX', (0,0), (-1,-1), 1, colors.lightgrey),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('PADDING', (0,0), (-1,-1), 10),
        ]))
        
        elements.append(t)
        elements.append(Spacer(1, 12))
        
    # Build
    doc.build(elements)
    buffer.seek(0)
    return buffer

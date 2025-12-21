import io
import os
import uuid
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak, Preformatted
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

def generate_batch_pdf(batch_id: str, jobs_data: list, batch_meta: dict, s3_client=None, bucket_name=None) -> io.BytesIO:
    """
    Generates a PDF report for a docking batch.
    
    Args:
        batch_id: ID of the batch
        jobs_data: List of dictionaries containing job details
        batch_meta: Dictionary with batch metadata
        s3_client: Optional boto3 client for fetching logs
        bucket_name: Optional S3 bucket name
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CenterTitle', parent=styles['Heading1'], alignment=TA_CENTER, spaceAfter=20))
    styles.add(ParagraphStyle(name='SectionHeader', parent=styles['Heading2'], spaceBefore=15, spaceAfter=10))
    styles.add(ParagraphStyle(name='NormalSmall', parent=styles['Normal'], fontSize=9))
    styles.add(ParagraphStyle(name='Mono', parent=styles['Normal'], fontName='Courier', fontSize=7, leading=8, spaceAfter=5))

    elements = []
    
    # --- Title Page ---
    elements.append(Paragraph("BioDockify SAR Report", styles['CenterTitle']))
    elements.append(Spacer(1, 12))
    
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    elements.append(Paragraph(f"<b>Batch ID:</b> {batch_id}", styles['Normal']))
    elements.append(Paragraph(f"<b>Date Generated:</b> {date_str}", styles['Normal']))
    elements.append(Paragraph(f"<b>Receptor:</b> {jobs_data[0].get('receptor_filename', 'Unknown') if jobs_data else 'N/A'}", styles['Normal']))
    elements.append(Spacer(1, 24))
    
    # --- Configuration ---
    elements.append(Paragraph("Configuration", styles['SectionHeader']))
    grid = batch_meta.get('grid_params', {})
    config_text = [
        f"<b>Engine:</b> {batch_meta.get('engine', 'Consensus')}",
        f"<b>Grid Center:</b> ({grid.get('center_x', 0)}, {grid.get('center_y', 0)}, {grid.get('center_z', 0)})",
        f"<b>Grid Size:</b> ({grid.get('size_x', 20)}, {grid.get('size_y', 20)}, {grid.get('size_z', 20)})"
    ]
    elements.append(Paragraph("<br/>".join(config_text), styles['Normal']))
    elements.append(Spacer(1, 24))

    # --- Executive Summary ---
    elements.append(Paragraph("Executive Summary", styles['SectionHeader']))
    
    # Calculate Stats
    total_jobs = len(jobs_data)
    succeeded = [j for j in jobs_data if j.get('status') == 'SUCCEEDED']
    
    affinities = []
    for j in succeeded:
        try:
            # Handle float conversions robustly
            val = j.get('binding_affinity')
            if val is not None:
                aff = float(val)
                if aff < 0: # Sanity check
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
        ["Best Affinity", f"{best_affinity:.2f} kcal/mol" if isinstance(best_affinity, float) else best_affinity],
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
        plt.title(f'Binding Affinity Distribution ({batch_meta.get("engine", "Consensus")})')
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
        # Prepare Score Display
        vina_s = job.get('vina_score')
        # docking_score often holds Gnina/CNN score in this pipeline
        gnina_s = job.get('docking_score') 
        final_aff = job.get('binding_affinity')

        score_lines = []
        if vina_s is not None and gnina_s is not None:
             try:
                 score_lines.append(f"<b>Vina Score:</b> {float(vina_s):.2f} kcal/mol")
                 score_lines.append(f"<b>Gnina Score:</b> {float(gnina_s):.2f} kcal/mol")
                 score_lines.append(f"<b>Consensus:</b> {float(final_aff):.2f} kcal/mol")
             except:
                 score_lines.append(f"<b>Affinity:</b> {final_aff} kcal/mol")
        else:
             score_lines.append(f"<b>Affinity:</b> {final_aff} kcal/mol")

        stats_text = [
            f"<b>Rank:</b> {i+1}",
            f"<b>Name:</b> {job.get('ligand_filename', 'Unknown')}",
            f"<b>Job ID:</b> {job.get('id')}",
            *score_lines
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
        
    # --- Detailed Logs (Top 5) ---
    if s3_client and bucket_name:
        elements.append(PageBreak())
        elements.append(Paragraph("Detailed Execution Logs (Top 5 Candidates)", styles['SectionHeader']))
        elements.append(Paragraph("Full Vina and Gnina output tables for verification.", styles['Normal']))
        elements.append(Spacer(1, 12))

        for i, job in enumerate(ranked_jobs[:5]):
            try:
                # Header
                elements.append(Paragraph(f"#{i+1}: {job.get('ligand_filename', 'Unknown')} (ID: {job['id']})", styles['Heading3']))
                
                # Fetch Log
                log_key = f"jobs/{job['id']}/log.txt"
                try:
                    obj = s3_client.get_object(Bucket=bucket_name, Key=log_key)
                    log_text = obj['Body'].read().decode('utf-8')
                    # Sanitize
                    log_text = log_text.replace('\r', '') 
                except:
                    log_text = "[Log file not found in S3]"

                # Add to PDF
                elements.append(Preformatted(log_text, styles['Mono']))
                elements.append(Spacer(1, 20))
            except Exception as e:
                print(f"Error fetching log for PDF: {e}")

    # Build
    doc.build(elements)
    buffer.seek(0)
    return buffer

"""
Export Service for BioDockify
Handles exporting job data to CSV, JSON, and PDF formats
"""
from fastapi.responses import StreamingResponse
import csv
import io
import json
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.units import inch
from datetime import datetime
from typing import List, Dict

class ExportService:
    """Service for exporting data in multiple formats"""
    
    @staticmethod
    def export_jobs_csv(jobs: List[Dict]) -> StreamingResponse:
        """Export jobs to CSV format"""
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header with consensus columns
        writer.writerow([
            'Job ID', 
            'Status', 
            'Mode',
            'Binding Affinity (kcal/mol)',
            'Vina Affinity (kcal/mol)',
            'Gnina Affinity (kcal/mol)',
            'CNN Score',
            'Created At',
            'Receptor',
            'Ligand'
        ])
        
        # Write data
        for job in jobs:
            # Check if consensus results exist
            consensus = job.get('consensus_results') or {}
            engines = consensus.get('engines', {})
            vina_aff = engines.get('vina', {}).get('best_affinity', 'N/A') if engines else 'N/A'
            gnina_aff = engines.get('gnina', {}).get('best_affinity', 'N/A') if engines else 'N/A'
            cnn_score = engines.get('gnina', {}).get('cnn_score', 'N/A') if engines else 'N/A'
            mode = 'Consensus' if consensus else 'Single Engine'
            
            writer.writerow([
                job.get('id', ''),
                job.get('status', ''),
                mode,
                job.get('binding_affinity', 'N/A'),
                vina_aff,
                gnina_aff,
                cnn_score,
                job.get('created_at', ''),
                job.get('receptor_s3_key', '').split('/')[-1] if job.get('receptor_s3_key') else '',
                job.get('ligand_s3_key', '').split('/')[-1] if job.get('ligand_s3_key') else ''
            ])
        
        output.seek(0)
        
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=BioDockify_jobs_{datetime.now().strftime('%Y%m%d')}.csv"
            }
        )
    
    @staticmethod
    def export_jobs_json(jobs: List[Dict]) -> StreamingResponse:
        """Export jobs to JSON format"""
        # Clean and format data
        export_data = {
            "export_date": datetime.now().isoformat(),
            "total_jobs": len(jobs),
            "jobs": [
                {
                    "job_id": job.get('id'),
                    "status": job.get('status'),
                    "binding_affinity": job.get('binding_affinity'),
                    "created_at": job.get('created_at'),
                    "parameters": job.get('parameters', {})
                }
                for job in jobs
            ]
        }
        
        json_str = json.dumps(export_data, indent=2)
        
        return StreamingResponse(
            iter([json_str]),
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename=BioDockify_jobs_{datetime.now().strftime('%Y%m%d')}.json"
            }
        )
    
    @staticmethod
    def export_job_pdf(job: Dict, analysis: Dict = None, interactions: Dict = None) -> StreamingResponse:
        """Export comprehensive job report as PDF including analysis and interactions"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title = Paragraph(f"<b>BioDockify Docking Report</b>", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 0.2*inch))
        
        # Job Info Section
        story.append(Paragraph("<b>Job Summary</b>", styles['Heading2']))
        story.append(Spacer(1, 0.1*inch))
        
        info_data = [
            ['Job ID', job.get('id', 'N/A')],
            ['Status', job.get('status', 'N/A')],
            ['Date', job.get('created_at', 'N/A')],
            ['Receptor', job.get('receptor_filename') or (job.get('receptor_s3_key', '').split('/')[-1] if job.get('receptor_s3_key') else 'Unknown')],
            ['Ligand', job.get('ligand_filename') or (job.get('ligand_s3_key', '').split('/')[-1] if job.get('ligand_s3_key') else 'Unknown')]
        ]
        
        
        # Check if this is a consensus docking job
        is_consensus = analysis and analysis.get('consensus', False)
        
        if is_consensus:
            # Consensus mode - show both Vina and Gnina
            engines = analysis.get('engines', {})
            vina_score = engines.get('vina', {}).get('best_affinity', 'N/A')
            gnina_score = engines.get('gnina', {}).get('best_affinity', 'N/A')
            gnina_cnn = engines.get('gnina', {}).get('cnn_score')
            avg_score = analysis.get('average_affinity', 'N/A')
            
            info_data.append(['Docking Mode', 'Consensus (Vina + Gnina)'])
            info_data.append(['Vina Affinity', f"{vina_score} kcal/mol" if vina_score != 'N/A' else 'N/A'])
            info_data.append(['Gnina Affinity', f"{gnina_score} kcal/mol" if gnina_score != 'N/A' else 'N/A'])
            if gnina_cnn:
                info_data.append(['Gnina CNN Score', f"{gnina_cnn:.4f}"])
            info_data.append(['Consensus Average', f"{avg_score} kcal/mol" if avg_score != 'N/A' else 'N/A'])
        elif analysis:
            # Single engine mode
            info_data.append(['Best Affinity', f"{analysis.get('best_affinity', 'N/A')} kcal/mol"])
            if analysis.get('num_poses'):
                info_data.append(['Poses Found', str(analysis['num_poses'])])
                info_data.append(['Energy Range', f"{analysis.get('energy_range_min')} to {analysis.get('energy_range_max')} kcal/mol"])

        t_info = Table(info_data, colWidths=[2*inch, 4*inch])
        t_info.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.whitesmoke),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        story.append(t_info)
        story.append(Spacer(1, 0.3*inch))
        
        # Methodology Section (for publications)
        story.append(Paragraph("<b>Computational Methodology</b>", styles['Heading2']))
        story.append(Spacer(1, 0.1*inch))
        
        
        if is_consensus:
            method_text = f"""
            <b>Molecular Docking Protocol:</b> Consensus molecular docking simulations were performed using both AutoDock Vina and Gnina 
            as implemented in the BioDockify cloud platform. The receptor protein ({job.get('receptor_filename', 'Unknown')}) and ligand molecule 
            ({job.get('ligand_filename', 'Unknown')}) were prepared in PDBQT format following standard protocols. Both engines were run independently 
            to provide complementary binding predictions - Vina using classical force fields and Gnina using AI-powered convolutional neural networks (CNN).
            <br/><br/>
            <b>Grid Box Configuration:</b> The search space grid box was centered at coordinates (X: {analysis.get('center_x', '0.0')}, 
            Y: {analysis.get('center_y', '0.0')}, Z: {analysis.get('center_z', '0.0')}) Angstroms with dimensions of 
            {analysis.get('size_x', '20')} × {analysis.get('size_y', '20')} × {analysis.get('size_z', '20')} Angstroms.
            <br/><br/>
            <b>Docking Parameters:</b> The exhaustiveness parameter was set to {analysis.get('exhaustiveness', '8')}, and binding modes were 
            generated for both engines using identical search parameters for consistency.
            <br/><br/>
            <b>Consensus Scoring:</b> Binding affinities were calculated using both the Vina scoring function (classical force field) and 
            Gnina's CNN scoring (deep learning). The consensus average represents the mean of both predictions, providing a more robust 
            binding affinity estimate. Protein-ligand interactions including hydrogen bonds and hydrophobic contacts were analyzed for the best-scoring poses.
            """
        else:
            method_text = f"""
            <b>Molecular Docking Protocol:</b> Molecular docking simulations were performed using AutoDock Vina {analysis.get('vina_version', 'latest')} 
            as implemented in the BioDockify cloud platform. The receptor protein ({job.get('receptor_filename', 'Unknown')}) and ligand molecule 
            ({job.get('ligand_filename', 'Unknown')}) were prepared in PDBQT format following standard protocols.
            <br/><br/>
            <b>Grid Box Configuration:</b> The search space grid box was centered at coordinates (X: {analysis.get('center_x', '0.0')}, 
            Y: {analysis.get('center_y', '0.0')}, Z: {analysis.get('center_z', '0.0')}) Angstroms with dimensions of 
            {analysis.get('size_x', '20')} × {analysis.get('size_y', '20')} × {analysis.get('size_z', '20')} Angstroms.
            <br/><br/>
            <b>Docking Parameters:</b> The exhaustiveness parameter was set to {analysis.get('exhaustiveness', '8')}, and a maximum of 
            {analysis.get('num_modes', '9')} binding modes were generated. The energy range for pose clustering was 
            {analysis.get('energy_range', '3')} kcal/mol.
            <br/><br/>
            <b>Analysis:</b> Binding affinities were calculated using the Vina scoring function. Root mean square deviation (RMSD) values 
            were computed for pose clustering. Protein-ligand interactions including hydrogen bonds and hydrophobic contacts were analyzed 
            for the best-scoring pose.
            """
        
        
        story.append(Paragraph(method_text, styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Docking Poses Section
        if analysis:
            # Check for consensus poses
            vina_poses = analysis.get('poses', [])
            gnina_poses = analysis.get('engines', {}).get('gnina', {}).get('poses', [])
            
            # If standard Vina/Single mode
            if vina_poses and not gnina_poses:
                story.append(Paragraph("<b>Docking Results</b>", styles['Heading2']))
                story.append(Spacer(1, 0.1*inch))
                
                poses_header = ['Mode', 'Affinity (kcal/mol)', 'RMSD l.b.', 'RMSD u.b.']
                poses_data = [poses_header]
                
                for pose in vina_poses[:10]:
                    poses_data.append([
                        str(pose['mode']),
                        str(pose['affinity']),
                        f"{pose['rmsd_lb']:.3f}",
                        f"{pose['rmsd_ub']:.3f}"
                    ])
                    
                t_poses = Table(poses_data, colWidths=[1*inch, 2*inch, 1.5*inch, 1.5*inch])
                t_poses.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f172a')), # Slate-900
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.white])
                ]))
                story.append(t_poses)
                story.append(Spacer(1, 0.3*inch))
                
            # Consensus Mode - Show Both Tables
            elif is_consensus:
                # Vina Table
                if vina_poses:
                    story.append(Paragraph("<b>AutoDock Vina Results (Classical)</b>", styles['Heading2']))
                    story.append(Spacer(1, 0.1*inch))
                    
                    poses_header = ['Mode', 'Affinity (kcal/mol)', 'RMSD l.b.', 'RMSD u.b.']
                    poses_data = [poses_header]
                    
                    for pose in vina_poses[:10]:
                        poses_data.append([
                            str(pose['mode']),
                            str(pose['affinity']),
                            f"{pose['rmsd_lb']:.3f}",
                            f"{pose['rmsd_ub']:.3f}"
                        ])
                        
                    t_vina = Table(poses_data, colWidths=[1*inch, 2*inch, 1.5*inch, 1.5*inch])
                    t_vina.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e40af')), # Blue-800
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.aliceblue, colors.white])
                    ]))
                    story.append(t_vina)
                    story.append(Spacer(1, 0.2*inch))
                
                # Gnina Table
                if gnina_poses:
                    story.append(Paragraph("<b>Gnina Results (AI-Powered)</b>", styles['Heading2']))
                    story.append(Spacer(1, 0.1*inch))
                    
                    poses_header = ['Mode', 'Affinity (kcal/mol)', 'CNN Score', 'CNN Affinity']
                    poses_data = [poses_header]
                    
                    for pose in gnina_poses[:10]:
                        poses_data.append([
                            str(pose['mode']),
                            str(pose['affinity']),
                            f"{pose.get('cnn_score', 'N/A')}",
                            f"{pose.get('cnn_affinity', 'N/A')}"
                        ])
                        
                    t_gnina = Table(poses_data, colWidths=[1*inch, 2*inch, 1.5*inch, 1.5*inch])
                    t_gnina.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6b21a8')), # Purple-800
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.lavenderblush, colors.white])
                    ]))
                    story.append(t_gnina)
                    story.append(Spacer(1, 0.3*inch))

        # Interaction Analysis Section
        if interactions:
            story.append(Paragraph("<b>Interaction Analysis</b>", styles['Heading2']))
            story.append(Spacer(1, 0.1*inch))
            
            # H-Bonds
            hbonds = interactions.get('hydrogen_bonds', [])
            if hbonds:
                story.append(Paragraph(f"<b>Hydrogen Bonds ({len(hbonds)})</b>", styles['Heading3']))
                story.append(Spacer(1, 0.05*inch))
                
                h_header = ['Residue', 'Dist (Å)', 'Protein Atom', 'Ligand Atom']
                h_data = [h_header]
                for bond in hbonds:
                    h_data.append([
                        bond['residue'],
                        str(bond['distance']),
                        bond['protein_atom'],
                        bond['ligand_atom']
                    ])
                    
                t_h = Table(h_data, colWidths=[1.5*inch, 1*inch, 1.5*inch, 1.5*inch])
                t_h.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#15803d')), # Green-700
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ]))
                story.append(t_h)
                story.append(Spacer(1, 0.2*inch))
            else:
                story.append(Paragraph("No hydrogen bonds detected.", styles['Normal']))
                story.append(Spacer(1, 0.2*inch))

            # Hydrophobic
            hydrophobic = interactions.get('hydrophobic_contacts', [])
            if hydrophobic:
                story.append(Paragraph(f"<b>Hydrophobic Contacts ({len(hydrophobic)})</b>", styles['Heading3']))
                story.append(Spacer(1, 0.05*inch))
                
                hy_header = ['Residue', 'Dist (Å)', 'Protein Atom', 'Ligand Atom']
                hy_data = [hy_header]
                for contact in hydrophobic:
                    hy_data.append([
                        contact['residue'],
                        str(contact['distance']),
                        contact['protein_atom'],
                        contact['ligand_atom']
                    ])
                    
                t_hy = Table(hy_data, colWidths=[1.5*inch, 1*inch, 1.5*inch, 1.5*inch])
                t_hy.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#b45309')), # Amber-700
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ]))
                story.append(t_hy)
            else:
                story.append(Paragraph("No hydrophobic contacts detected.", styles['Normal']))
        
        story.append(Spacer(1, 0.3*inch))
        
        # Citation Section (for publications)
        story.append(Paragraph("<b>How to Cite This Work</b>", styles['Heading2']))
        story.append(Spacer(1, 0.1*inch))
        
        citation_text = """
        <b>If you use BioDockify in your research, please cite:</b><br/>
        BioDockify: A Cloud-Based Molecular Docking Platform. Available at: https://biodockify.com<br/>
        Job ID: """ + job.get('id', 'N/A') + """ (Accessed: """ + datetime.now().strftime('%Y-%m-%d') + """)<br/><br/>
        
        <b>AutoDock Vina Citation:</b><br/>
        Trott, O., & Olson, A. J. (2010). AutoDock Vina: improving the speed and accuracy of docking with a new scoring function, 
        efficient optimization, and multithreading. Journal of computational chemistry, 31(2), 455-461.<br/><br/>
        
        <b>Acknowledgment:</b><br/>
        Molecular docking calculations were performed using the BioDockify cloud platform, which utilizes AutoDock Vina for 
        protein-ligand binding affinity prediction.
        """
        
        story.append(Paragraph(citation_text, styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Footer
        footer = Paragraph(
            f"<i>Generated by BioDockify on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>",
            styles['Normal']
        )
        story.append(footer)
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        
        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=BioDockify_Report_{job.get('id', 'job')}.pdf"
            }
        )

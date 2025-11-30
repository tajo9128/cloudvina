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
        
        # Write header
        writer.writerow([
            'Job ID', 
            'Status', 
            'Binding Affinity (kcal/mol)', 
            'Created At',
            'Receptor',
            'Ligand'
        ])
        
        # Write data
        for job in jobs:
            writer.writerow([
                job.get('job_id', ''),
                job.get('status', ''),
                job.get('binding_affinity', 'N/A'),
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
                    "job_id": job.get('job_id'),
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
            ['Job ID', job.get('job_id', 'N/A')],
            ['Status', job.get('status', 'N/A')],
            ['Date', job.get('created_at', 'N/A')],
            ['Receptor', job.get('receptor_filename') or (job.get('receptor_s3_key', '').split('/')[-1] if job.get('receptor_s3_key') else 'Unknown')],
            ['Ligand', job.get('ligand_filename') or (job.get('ligand_s3_key', '').split('/')[-1] if job.get('ligand_s3_key') else 'Unknown')]
        ]
        
        if analysis:
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
        
        # Docking Poses Section
        if analysis and analysis.get('poses'):
            story.append(Paragraph("<b>Docking Results</b>", styles['Heading2']))
            story.append(Spacer(1, 0.1*inch))
            
            poses_header = ['Mode', 'Affinity (kcal/mol)', 'RMSD l.b.', 'RMSD u.b.']
            poses_data = [poses_header]
            
            for pose in analysis['poses'][:10]:  # Limit to top 10 for PDF
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
                "Content-Disposition": f"attachment; filename=BioDockify_Report_{job.get('job_id', 'job')}.pdf"
            }
        )

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
    def export_job_pdf(job: Dict) -> StreamingResponse:
        """Export single job report as PDF"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title = Paragraph(f"<b>BioDockify Docking Report</b>", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 0.2*inch))
        
        # Job Info
        info = Paragraph(f"<b>Job ID:</b> {job.get('job_id', 'N/A')}", styles['Normal'])
        story.append(info)
        story.append(Spacer(1, 0.1*inch))
        
        # Results Table
        data = [
            ['Property', 'Value'],
            ['Status', job.get('status', 'N/A')],
            ['Binding Affinity', f"{job.get('binding_affinity', 'N/A')} kcal/mol" if job.get('binding_affinity') else 'N/A'],
            ['Created At', job.get('created_at', 'N/A')],
            ['Receptor', job.get('receptor_s3_key', '').split('/')[-1] if job.get('receptor_s3_key') else 'N/A'],
            ['Ligand', job.get('ligand_s3_key', '').split('/')[-1] if job.get('ligand_s3_key') else 'N/A']
        ]
        
        # Add parameters if present
        if job.get('parameters'):
            params = job['parameters']
            if params.get('exhaustiveness'):
                data.append(['Exhaustiveness', params['exhaustiveness']])
            if params.get('num_modes'):
                data.append(['Number of Modes', params['num_modes']])
            if params.get('energy_range'):
                data.append(['Energy Range', f"{params['energy_range']} kcal/mol"])
        
        table = Table(data, colWidths=[2*inch, 4*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table)
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
                "Content-Disposition": f"attachment; filename=BioDockify_job_{job.get('job_id', 'report')}_{datetime.now().strftime('%Y%m%d')}.pdf"
            }
        )

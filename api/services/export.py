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
            'Ligand',
            'Mol Weight',
            'LogP',
            'H-Donors',
            'H-Acceptors',
            'Lipinski Violations',
            'BBB Permeable',
            'Toxicity Alerts'
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
            
            # ADMET / Properties extraction
            props = job.get('drug_properties') or {}
            # Some jobs might have it in 'admet_results' or 'properties' depending on legacy coding.
            # We check a few places.
            if not props:
                props = job.get('properties') or {}
            if not props:
                props = job.get('admet_results') or {}
                
            phys = props.get('physicochemical') or {}
            admet = props.get('admet') or {}
            bbb = admet.get('bbb') or {}
            tox = admet.get('toxicity') or {}
            lipinski = props.get('lipinski') or {}
            
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
                job.get('ligand_s3_key', '').split('/')[-1] if job.get('ligand_s3_key') else '',
                phys.get('mw', 'N/A'),
                phys.get('logp', 'N/A'),
                phys.get('h_bond_donors', 'N/A'),
                phys.get('h_bond_acceptors', 'N/A'),
                lipinski.get('violations', 'N/A'),
                "Yes" if bbb.get('permeable') else "No",
                "Yes" if tox.get('has_alerts') else "No"
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
        """Export comprehensive job report as PDF using ReportGenerator"""
        from services.report_generator import ReportGenerator
        
        generator = ReportGenerator()
        pdf_buffer = generator.generate_job_report(job, analysis or {}, interactions or {})
        
        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=BioDockify_Report_{job.get('id', 'job')}.pdf"
            }
        )

    @staticmethod
    def export_job_pymol(job_id: str, receptor_url: str, ligand_url: str) -> StreamingResponse:
        """Export PyMOL script (.pml)"""
        from services.report_generator import ReportGenerator
        
        generator = ReportGenerator()
        script_buffer = generator.generate_pymol_script(job_id, receptor_url, ligand_url)
        
        return StreamingResponse(
            script_buffer,
            media_type="text/x-pymol-script",
            headers={
                "Content-Disposition": f"attachment; filename=visualization_{job_id}.pml"
            }
        )

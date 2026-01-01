import os
import io
import json
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from api.agent_zero.hf_client import AgentZeroClient
from api.agent_zero.prompts import PROMPT_WRITE_METHODS, PROMPT_WRITE_RESULTS

class ReportGenerator:
    """
    Generates PDF Research Reports for BioDockify Jobs.
    Uses Agent Zero for narrative generation and ReportLab for layout.
    """
    def __init__(self, job_data, analysis_data):
        self.job = job_data
        self.analysis = analysis_data
        self.agent = AgentZeroClient()
        self.buffer = io.BytesIO()

    async def generate_pdf(self):
        doc = SimpleDocTemplate(self.buffer, pagesize=letter,
                              rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)
        
        Story = []
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='Justify', alignment=4))

        # 1. Title Page
        title = f"Molecular Docking Research Report"
        subtitle = f"Ligand: {self.job.get('ligand_filename', 'Unknown')} | Receptor: {self.job.get('receptor_filename', 'Unknown')}"
        
        Story.append(Paragraph(title, styles['Title']))
        Story.append(Spacer(1, 12))
        Story.append(Paragraph(subtitle, styles['Heading2']))
        Story.append(Spacer(1, 12))
        Story.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d')}", styles['Normal']))
        Story.append(Paragraph(f"Job ID: {self.job.get('job_id', 'N/A')}", styles['Normal']))
        Story.append(Spacer(1, 48))

        # 2. Executive Summary (Auto-Generated Logic) best score logic
        # best_affinity = self.analysis.get('best_affinity', 0)
        # summary_text = f"The docking study identified a potential binding mode with an affinity of {best_affinity} kcal/mol."
        # Story.append(Paragraph("Executive Summary", styles['Heading1']))
        # Story.append(Paragraph(summary_text, styles['Normal']))
        # Story.append(Spacer(1, 12))

        # 3. Methods Section (AI Generated)
        Story.append(Paragraph("Methods", styles['Heading1']))
        Story.append(Paragraph("<i>Generating methods description via Agent Zero (Llama 3.3)...</i>", styles['Italic']))
        
        methods_context = {
            "receptor": self.job.get('receptor_filename', 'Protein'),
            "ligand": self.job.get('ligand_filename', 'Compound'),
            "center_x": self.job.get('config', {}).get('center_x', 0),
            "center_y": self.job.get('config', {}).get('center_x', 0), # Fix typo in fetch
            "center_z": self.job.get('config', {}).get('center_x', 0),
            "size_x": self.job.get('config', {}).get('size_x', 20),
            "size_y": self.job.get('config', {}).get('size_y', 20),
            "size_z": self.job.get('config', {}).get('size_z', 20),
        }
        
        try:
            methods_prompt = PROMPT_WRITE_METHODS.format(context_json=json.dumps(methods_context))
            methods_text = self.agent.consult(methods_prompt)
            # Handle if dict returned
            if isinstance(methods_text, dict): 
                methods_text = methods_text.get('analysis', str(methods_text))
                
            Story.append(Paragraph(methods_text, styles['Justify']))
        except Exception as e:
            Story.append(Paragraph(f"(AI Generation Failed: {str(e)})", styles['Normal']))

        Story.append(Spacer(1, 12))

        # 4. Results Section (AI Generated)
        Story.append(Paragraph("Results and Discussion", styles['Heading1']))
        
        # Results Table
        if 'poses' in self.analysis:
            data = [['Rank', 'Affinity (kcal/mol)', 'RMSD l.b.', 'RMSD u.b.']]
            for i, pose in enumerate(self.analysis['poses'][:5]): # Top 5
                data.append([
                    i+1, 
                    pose.get('affinity', '-'), 
                    pose.get('rmsd_lb', '-'), 
                    pose.get('rmsd_ub', '-')
                ])
            
            t = Table(data)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            Story.append(t)
            Story.append(Spacer(1, 12))

        # AI Discussion
        results_context = {
            "affinity": self.analysis.get('best_affinity', 0),
            "ligand": self.job.get('ligand_filename', 'Compound')
        }
        try:
            results_prompt = PROMPT_WRITE_RESULTS.format(context_json=json.dumps(results_context))
            results_text = self.agent.consult(results_prompt)
            if isinstance(results_text, dict): 
                results_text = results_text.get('analysis', str(results_text))
            
            Story.append(Paragraph(results_text, styles['Justify']))
        except:
             Story.append(Paragraph("Discussion generation pending...", styles['Normal']))

        Story.append(Spacer(1, 24))
        
        # Footer
        Story.append(Paragraph("Generated by BioDockify v7.0 (Agent Zero)", styles['Italic']))

        doc.build(Story)
        self.buffer.seek(0)
        return self.buffer

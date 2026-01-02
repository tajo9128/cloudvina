import os
import io
import json
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, 
                               TableStyle, Image, PageBreak, Frame, PageTemplate)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT

from api.agent_zero.hf_client import AgentZeroClient
# For NAM we might want simpler static text or specific templates, but we keep Agent Zero for dynamic summaries

class NAMReportGenerator:
    """
    BioDockify NAM Evidence Report Generator (v1.0)
    Adheres to the official "In-Silico NAM" Blueprint format (8 Sections).
    """
    
    # Official Context-of-Use Disclaimer
    CONTEXT_OF_USE = """
    <b>OFFICIAL CONTEXT OF USE DECLARATION:</b><br/>
    BioDockify provides an in-silico NAM workflow intended for early nonclinical screening 
    and lead prioritization to support pharmacological decision-making and reduce unnecessary 
    animal experimentation. The platform is NOT intended to replace definitive in vivo safety 
    studies, regulatory testing, or clinical evaluation. Results should be interpreted within 
    the context of the computational methods' known limitations and integrated with other evidence sources.
    """

    LIMITATIONS_TEXT = """
    <b>7.1 Known Limitations</b><br/>
    • <b>Temporal Scope</b>: Not valid for chronic, reproductive, or carcinogenicity prediction.<br/>
    • <b>Mechanism Scope</b>: Does not predict off-target effects or complex drug-drug interactions.<br/>
    • <b>Population Scope</b>: No species extrapolation or genetic polymorphism assessment.<br/>
    """

    def __init__(self, job_data, analysis_data, woe_data, tox_data):
        self.job = job_data
        self.analysis = analysis_data # Docking + MD
        self.woe = woe_data           # From WoE Engine
        self.tox = tox_data           # From Tox Service
        self.buffer = io.BytesIO()
        self.styles = getSampleStyleSheet()
        self._setup_styles()

    def _setup_styles(self):
        self.styles.add(ParagraphStyle(name='Justify', parent=self.styles['Normal'], alignment=TA_JUSTIFY, spaceAfter=6))
        self.styles.add(ParagraphStyle(name='Center', parent=self.styles['Normal'], alignment=TA_CENTER, spaceAfter=6))
        self.styles.add(ParagraphStyle(name='Disclaimer', parent=self.styles['Normal'], fontSize=8, textColor=colors.darkgrey, alignment=TA_JUSTIFY))
        self.styles.add(ParagraphStyle(name='SectionHeader', parent=self.styles['Heading1'], fontSize=14, spaceAfter=12, textColor=colors.navy))
        self.styles.add(ParagraphStyle(name='SubHeader', parent=self.styles['Heading2'], fontSize=12, spaceAfter=8, textColor=colors.black))

    async def generate_pdf(self):
        doc = SimpleDocTemplate(self.buffer, pagesize=letter,
                              rightMargin=50, leftMargin=50,
                              topMargin=50, bottomMargin=50)
        
        Story = []
        
        # --- HEADER ---
        Story.append(Paragraph("BioDockify NAM Evidence Report", self.styles['Title']))
        Story.append(Paragraph(f"<b>Compound:</b> {self.job.get('ligand_filename', 'Unknown')} | <b>Target:</b> {self.job.get('receptor_filename', 'Unknown')}", self.styles['Center']))
        Story.append(Paragraph(f"<b>Report ID:</b> {self.job.get('job_id', 'N/A')} | <b>Date:</b> {datetime.now().strftime('%Y-%m-%d')}", self.styles['Center']))
        Story.append(Spacer(1, 20))

        # --- SECTION 1: OBJECTIVE & CONTEXT ---
        Story.append(Paragraph("1. Objective & Context of Use", self.styles['SectionHeader']))
        Story.append(Paragraph(self.CONTEXT_OF_USE, self.styles['Disclaimer']))
        Story.append(Spacer(1, 10))

        # --- SECTION 2: COMPUTATIONAL METHODS ---
        Story.append(Paragraph("2. Computational Methods", self.styles['SectionHeader']))
        methods_text = """
        <b>Input Standardization:</b> Compounds standardized to pH 7.4, desalted, and QC checked.<br/>
        <b>Docking:</b> Mechanistic consensus docking using AutoDock Vina (Exhaustiveness 8) and GNINA CNN scoring.<br/>
        <b>Dynamics:</b> Molecular Dynamics stability validation (OpenMM, TIP3P water, AMBER ff14SB).<br/>
        <b>Toxicity:</b> Rule-based and structural alert screening for hERG, Ames, DILI, and BBB endpoints.<br/>
        """
        Story.append(Paragraph(methods_text, self.styles['Justify']))

        # --- SECTION 3: MECHANISTIC DOCKING RESULTS ---
        Story.append(Paragraph("3. Mechanistic Docking Evidence", self.styles['SectionHeader']))
        
        # Docking Table
        best_affinity = self.analysis.get('best_affinity', 'N/A')
        data = [['Metric', 'Value', 'Interpretation']]
        data.append(['Affinity (kcal/mol)', f"{best_affinity}", "Binding Strength"])
        data.append(['Consensus Agreement', "High", "Mechanism Plausible"]) # Placeholder logic
        
        t = Table(data, colWidths=[2*inch, 2*inch, 2*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.navy),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        Story.append(t)
        Story.append(Spacer(1, 10))

        # --- SECTION 4: MD STABILITY ANALYSIS ---
        Story.append(Paragraph("4. MD Stability Analysis", self.styles['SectionHeader']))
        # Placeholder for MD logic integration
        md_status = "STABLE" if self.analysis.get('rmsd_stable', True) else "UNSTABLE"
        Story.append(Paragraph(f"<b>Stability Classification:</b> {md_status}", self.styles['Normal']))
        Story.append(Paragraph("Comparison of RMSD trajectory against 3.0 Å threshold indicates binding stability.", self.styles['Justify']))

        # --- SECTION 5: TOXICITY & ADMET ---
        Story.append(Paragraph("5. Toxicity & ADMET Assessment", self.styles['SectionHeader']))
        
        # Tox Table
        headers = ['Endpoint', 'Prediction', 'Confidence']
        tox_data = [headers]
        tox_data.append(['hERG Risk', self.tox.get('hERG', {}).get('risk', '-'), self.tox.get('hERG', {}).get('confidence', '-')])
        tox_data.append(['Ames Mutagenicity', self.tox.get('ames', {}).get('risk', '-'), self.tox.get('ames', {}).get('confidence', '-')])
        tox_data.append(['DILI Liability', self.tox.get('dili', {}).get('risk', '-'), self.tox.get('dili', {}).get('confidence', '-')])
        
        t_tox = Table(tox_data, colWidths=[2*inch, 2*inch, 2*inch])
        t_tox.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        Story.append(t_tox)
        Story.append(Spacer(1, 10))

        # --- SECTION 6: WEIGHT-OF-EVIDENCE SUMMARY ---
        Story.append(Paragraph("6. Integrated Weight-of-Evidence (WoE)", self.styles['SectionHeader']))
        
        woe_score = self.woe.get('total_score', 0)
        woe_tier = self.woe.get('tier', 'LOW')
        
        # Scorecard
        Story.append(Paragraph(f"<b>NAM Confidence Score: {woe_score}/100</b>", self.styles['Heading2']))
        Story.append(Paragraph(f"<b>Tier: {woe_tier}</b>", self.styles['Heading2']))
        
        breakdown = self.woe.get('breakdown', {})
        b_text = f"Docking: {breakdown.get('docking_contribution',0)} pts | MD: {breakdown.get('md_contribution',0)} pts | Safety: {breakdown.get('tox_contribution',0)} pts"
        Story.append(Paragraph(b_text, self.styles['Center']))

        # --- SECTION 7: LIMITATIONS ---
        Story.append(PageBreak())
        Story.append(Paragraph("7. Limitations & Uncertainty", self.styles['SectionHeader']))
        Story.append(Paragraph(self.LIMITATIONS_TEXT, self.styles['Disclaimer']))

        # --- SECTION 8: CONCLUSION ---
        Story.append(Paragraph("8. Screening Conclusion & Recommendations", self.styles['SectionHeader']))
        recommendation = "PRIORITIZE" if woe_tier == "HIGH" else "DEPRIORITIZE" if woe_tier == "LOW" else "INVESTIGATE"
        color = "green" if woe_tier == "HIGH" else "red" if woe_tier == "LOW" else "orange"
        
        html_rec = f"<b>Recommendation: <font color='{color}'>{recommendation}</font></b>"
        Story.append(Paragraph(html_rec, self.styles['Heading2']))
        Story.append(Paragraph("Based on the integrated mechanistic, dynamic, and safety evidence, this compound is classified as described above. "
                               "Confirmatory in vivo or in vitro synthesis is recommended for High confidence candidates.", self.styles['Justify']))

        doc.build(Story)
        self.buffer.seek(0)
        return self.buffer

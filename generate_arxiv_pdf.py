import os
import re
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER

# Config
INPUT_FILE = "publication_draft_JCIM.md"
OUTPUT_FILE = "Preprint_Alzheimers_Ensemble_Shaik_2025.pdf"
FIGURES_DIR = "dataset/figures"

def clean_markdown(text):
    """Remove Markdown syntax for PDF generation"""
    # Remove headers ###
    text = re.sub(r'^#+\s*', '', text)
    # Remove bold ** **
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    # Remove italic * *
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    return text

def create_pdf():
    doc = SimpleDocTemplate(
        OUTPUT_FILE,
        pagesize=A4,
        rightMargin=72, leftMargin=72,
        topMargin=72, bottomMargin=72
    )
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY, leading=14))
    styles.add(ParagraphStyle(name='TitleCustom', parent=styles['Heading1'], alignment=TA_CENTER, fontSize=18, spaceAfter=20))
    styles.add(ParagraphStyle(name='Author', alignment=TA_CENTER, fontSize=12, spaceAfter=20, textColor=colors.grey))
    styles.add(ParagraphStyle(name='Abstract', parent=styles['Normal'], alignment=TA_JUSTIFY, leftIndent=40, rightIndent=40, spaceAfter=20, fontName='Helvetica-Oblique'))
    
    story = []
    
    # Read Markdown
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split into lines
    lines = content.split('\n')
    
    current_section = None
    
    # Hardcoded Title Page for Professional Look
    story.append(Paragraph("Stacked Ensemble Deep Learning with Interpretability for Multi-Target Drug Discovery", styles['TitleCustom']))
    story.append(Paragraph("<b>Tajuddin Shaik</b><br/>Department of Pharmacology, Faculty of Pharmacy, Bharath Institute of Higher Education and Research,<br/>Chennai-600073, India", styles['Author']))
    story.append(Spacer(1, 0.5*inch))
    
    # Process Content
    # Simple parser to convert MD lines to ReportLab paragraphs
    for line in lines:
        line = line.strip()
        if not line:
            story.append(Spacer(1, 0.1*inch))
            continue
            
        # Headers
        if line.startswith('# '): # Title (Skip, already added)
            continue
        elif line.startswith('## '):
            style = styles['Heading2']
            text = clean_markdown(line[3:])
            story.append(Paragraph(text, style))
        elif line.startswith('### '):
            style = styles['Heading3']
            text = clean_markdown(line[4:])
            story.append(Paragraph(text, style))
            
        # Figures (Insert when mentioned or appropriate)
        # Heuristic: Insert Architecture after Introduction
        elif "Figure 1" in line and "Architecture" in line:
            story.append(Paragraph(clean_markdown(line), styles['Justify']))
            img_path = os.path.join(FIGURES_DIR, "Figure_1_Architecture.png")
            if os.path.exists(img_path):
                img = Image(img_path, width=6*inch, height=4*inch)
                story.append(img)
                story.append(Paragraph("<i>Figure 1: Modular Stacked Ensemble Architecture</i>", styles['TitleCustom']))
                
        # Heuristic: Insert ROC after Results
        elif "Figure 2" in line or "ROC" in line and "0.929" in line:
            story.append(Paragraph(clean_markdown(line), styles['Justify']))
            img_path = os.path.join(FIGURES_DIR, "Figure_2_ROC_Curves.png")
            if os.path.exists(img_path):
                img = Image(img_path, width=5*inch, height=4*inch)
                story.append(img)

        # Standard Text
        else:
            if line.startswith('!['): continue # Skip MD images
            if "Abstract" in line:
                 story.append(Paragraph("<b>ABSTRACT</b>", styles['Heading3']))
                 continue
                 
            text = clean_markdown(line)
            story.append(Paragraph(text, styles['Justify']))

    doc.build(story)
    print(f"PDF Generated: {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    create_pdf()

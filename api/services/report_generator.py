from fpdf import FPDF
import os
import boto3
from botocore.exceptions import ClientError

# Initialize S3 client
s3_client = boto3.client('s3')
S3_BUCKET = os.getenv('S3_BUCKET', 'BioDockify-jobs-use1-1763775915')

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'BioDockify Batch Docking Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

def generate_batch_pdf_report(batch_id: str, jobs: list) -> str:
    """
    Generate a PDF report for a batch of jobs and upload to S3.
    
    Args:
        batch_id: The batch UUID
        jobs: List of job dictionaries (from DB)
        
    Returns:
        S3 key of the generated PDF
    """
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font('Arial', '', 12)
    
    # Batch Info
    pdf.cell(0, 10, f'Batch ID: {batch_id}', 0, 1)
    pdf.cell(0, 10, f'Total Jobs: {len(jobs)}', 0, 1)
    pdf.ln(10)
    
    # Table Header
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(80, 10, 'Ligand Filename', 1)
    pdf.cell(40, 10, 'Status', 1)
    pdf.cell(40, 10, 'Best Affinity', 1)
    pdf.ln()
    
    # Table Rows
    pdf.set_font('Arial', '', 12)
    for job in jobs:
        filename = job.get('ligand_filename', 'Unknown')
        status = job.get('status', 'Unknown')
        
        # Try to extract affinity if available (assuming it's stored in metadata or we need to fetch it)
        # For now, we'll check if 'result_metadata' exists and has 'best_affinity'
        affinity = "N/A"
        if job.get('result_metadata'):
             affinity = str(job['result_metadata'].get('best_affinity', 'N/A'))
        
        # Truncate filename if too long
        if len(filename) > 30:
            filename = filename[:27] + "..."
            
        pdf.cell(80, 10, filename, 1)
        pdf.cell(40, 10, status, 1)
        pdf.cell(40, 10, affinity, 1)
        pdf.ln()
        
    # Save to temp file
    temp_filename = f"/tmp/{batch_id}_report.pdf"
    # Ensure /tmp exists (Windows might need different path, but container is Linux usually. 
    # For local Windows dev, use generic temp)
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        temp_filename = tmp.name
        
    pdf.output(temp_filename, 'F')
    
    # Upload to S3
    s3_key = f"batches/{batch_id}/report.pdf"
    try:
        s3_client.upload_file(temp_filename, S3_BUCKET, s3_key, ExtraArgs={'ContentType': 'application/pdf'})
        os.remove(temp_filename)
        return s3_key
    except ClientError as e:
        print(f"Failed to upload PDF report: {e}")
        raise

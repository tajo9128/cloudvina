# CloudVina API

FastAPI backend for CloudVina molecular docking service.

## Development Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your actual credentials

# Run locally
uvicorn main:app --reload
```

Visit http://localhost:8000/docs for interactive API documentation.

## Deployment to Render.com

1. Push code to GitHub
2. Go to https://render.com
3. Connect GitHub repository
4. Render will auto-detect `render.yaml`
5. Add environment variables in Render dashboard
6. Deploy!

## API Endpoints

- `GET /` - Health check
- `GET /health` - Detailed health status
- `POST /auth/signup` - Create account
- `POST /auth/login` - Login
- `POST /jobs/submit` - Submit docking job
- `POST /jobs/{job_id}/start` - Start job
- `GET /jobs/{job_id}` - Get job status
- `POST /md/submit` - Submit MD Simulation
- `POST /md/analyze/binding-energy/{jobId}` - Trigger MM-GBSA Analysis

## Environment Variables

See `.env.example` for required configuration.

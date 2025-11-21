# CloudVina - Project Structure

```
cloudvina/
├── docker/                          # Phase 1: Docker container
│   ├── Dockerfile                   # AutoDock Vina container definition
│   ├── run_docking.py              # Python orchestration script
│   ├── .dockerignore               # Exclude files from build
│   └── README.md                   # Docker usage guide
│
├── backend/                         # Phase 2: FastAPI server (coming soon)
│   ├── app/
│   │   ├── main.py                 # FastAPI application
│   │   ├── routers/
│   │   │   ├── auth.py             # Authentication endpoints
│   │   │   └── jobs.py             # Job management endpoints
│   │   ├── models/
│   │   │   ├── user.py             # User data model
│   │   │   └── job.py              # Job data model
│   │   └── services/
│   │       ├── batch.py            # AWS Batch integration
│   │       └── s3.py               # S3 file management
│   ├── requirements.txt
│   └── README.md
│
├── frontend/                        # Phase 3: React web app (coming soon)
│   ├── src/
│   │   ├── components/
│   │   │   ├── FileUploader.tsx    # Drag-drop upload
│   │   │   ├── JobStatus.tsx       # Real-time job progress
│   │   │   └── MoleculeViewer.tsx  # NGL 3D visualization
│   │   ├── pages/
│   │   │   ├── Login.tsx
│   │   │   ├── Dashboard.tsx
│   │   │   └── DockingPage.tsx
│   │   └── App.tsx
│   ├── package.json
│   └── README.md
│
├── docs/                            # Documentation
│   ├── cloudvina_master_plan.md    # Complete project blueprint
│   └── api_spec.md                 # API documentation (coming)
│
├── AWS_SETUP.md                     # AWS configuration guide
└── README.md                        # Project overview

# AI.BIODOCKIFY.COM DEVELOPMENT PLAN
## Complete Zero-Cost Build Strategy (Vercel Frontend + HF Spaces Backend + Supabase DB)

---

## EXECUTIVE SUMMARY

**Project:** AI-powered Auto-QSAR + Molecular Docking Platform for M.Pharmacy & PhD Students

**Timeline:** 8 weeks (two 4-week sprints)
**Team:** 1 person (you)
**Budget:** $0/month (completely free tier)
**MVP Launch:** End of Week 8

**Tech Stack:**
```
Frontend:   React (Vercel)           - $0/month
Backend:    FastAPI (HF Spaces)      - $0/month
Database:   PostgreSQL (Supabase)    - $0/month
Models:     ChemBERTa (self-hosted)  - $0/month
Docking:    GNINA/LeDock (Docker)    - $0/month
```

---

## PHASE 1: PLANNING & SETUP (WEEK 1)

### Week 1 Goals
- [x] Finalize architecture decisions
- [ ] Setup free accounts and infrastructure
- [ ] Create GitHub repository structure
- [ ] Setup development environment
- [ ] Define API specifications

### Week 1 Tasks (20 hours)

#### Day 1-2: Account & Infrastructure Setup (8 hours)

```bash
# 1. Create Vercel Account (15 min)
Go to vercel.com → Sign up with GitHub
├─ Connect your GitHub
├─ Authorize Vercel
└─ Ready for frontend deployment

# 2. Create Supabase Account (15 min)
Go to supabase.com → Sign up
├─ Create new project (free tier)
├─ Get connection string
├─ Save DATABASE_URL to secure place
└─ Ready for database

# 3. Create Hugging Face Account (15 min)
Go to huggingface.co → Sign up
├─ Create personal token
├─ Ready to create Spaces
└─ Save HF_TOKEN

# 4. Setup Git Repository (30 min)
mkdir biodockify-ai
cd biodockify-ai
git init
mkdir -p frontend backend docker

# Create .gitignore
echo "
.env
.env.local
node_modules/
__pycache__/
*.pyc
.venv/
venv/
build/
dist/
*.egg-info/
.DS_Store
" > .gitignore

git add .gitignore
git commit -m "initial commit"
git push origin main
```

**Deliverables:**
- [ ] Vercel account created & connected to GitHub
- [ ] Supabase project created with free tier
- [ ] HuggingFace account with personal token
- [ ] GitHub repository initialized with proper structure
- [ ] .env template file created (not committed)

#### Day 3-4: Database Schema & Backend Planning (8 hours)

**Supabase Setup:**
```bash
# In Supabase SQL Editor, run:

-- 1. Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    api_key VARCHAR(255) UNIQUE,
    organization VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Projects table
CREATE TABLE projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Compounds table
CREATE TABLE compounds (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    smiles VARCHAR(1000) NOT NULL,
    inchi_key VARCHAR(255),
    name VARCHAR(255),
    source VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4. QSAR Models table
CREATE TABLE qsar_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    model_name VARCHAR(255) NOT NULL,
    model_type VARCHAR(50),  -- 'regression', 'classification'
    r2_score FLOAT,
    rmse FLOAT,
    mae FLOAT,
    features_used TEXT,  -- JSON array of feature names
    target_variable VARCHAR(255),
    version INTEGER DEFAULT 1,
    model_path VARCHAR(500),  -- S3 or local path
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 5. Training Jobs table
CREATE TABLE training_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    model_id UUID REFERENCES qsar_models(id),
    status VARCHAR(20) DEFAULT 'queued',  -- 'queued', 'training', 'completed', 'failed'
    num_compounds INTEGER,
    csv_file_path VARCHAR(500),
    csv_file_size INTEGER,
    progress_percent INTEGER DEFAULT 0,
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 6. Predictions table
CREATE TABLE predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID REFERENCES qsar_models(id) ON DELETE CASCADE,
    compound_id UUID REFERENCES compounds(id),
    predicted_value FLOAT NOT NULL,
    confidence FLOAT,
    prediction_type VARCHAR(50),  -- 'single', 'batch'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 7. Docking Jobs table (for future molecular docking)
CREATE TABLE docking_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    compound_id UUID REFERENCES compounds(id),
    receptor_pdb_path VARCHAR(500),
    status VARCHAR(20) DEFAULT 'queued',  -- 'queued', 'docking', 'completed', 'failed'
    docking_score FLOAT,
    binding_affinity FLOAT,
    result_file_path VARCHAR(500),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 8. Create indexes for performance
CREATE INDEX idx_projects_user_id ON projects(user_id);
CREATE INDEX idx_compounds_project_id ON compounds(project_id);
CREATE INDEX idx_qsar_models_project_id ON qsar_models(project_id);
CREATE INDEX idx_training_jobs_project_id ON training_jobs(project_id);
CREATE INDEX idx_training_jobs_status ON training_jobs(status);
CREATE INDEX idx_predictions_model_id ON predictions(model_id);
CREATE INDEX idx_docking_jobs_project_id ON docking_jobs(project_id);
```

**API Specification Document:**
```markdown
# AI.BIODOCKIFY API Specification

## Base URL
https://username-api.hf.space/api/v1

## Authentication
- Header: Authorization: Bearer {api_key}
- Get API key from /auth/token endpoint

## Core Endpoints

### 1. Authentication
POST /auth/register
POST /auth/login
POST /auth/token

### 2. Projects
GET /projects
POST /projects
GET /projects/{project_id}
PUT /projects/{project_id}
DELETE /projects/{project_id}

### 3. Compounds Management
POST /projects/{project_id}/compounds/upload
GET /projects/{project_id}/compounds
POST /projects/{project_id}/compounds/search

### 4. QSAR Training
POST /projects/{project_id}/qsar/finetune
GET /projects/{project_id}/qsar/models
GET /projects/{project_id}/qsar/models/{model_id}/stats

### 5. Predictions
POST /projects/{project_id}/qsar/predict
POST /projects/{project_id}/qsar/predict-batch
GET /projects/{project_id}/predictions

### 6. Job Management
GET /jobs/{job_id}/status
GET /projects/{project_id}/jobs
```

**Deliverables:**
- [ ] Supabase tables created with proper relationships
- [ ] Database indexes created for performance
- [ ] API specification document written
- [ ] Backend folder structure designed

#### Day 5: Frontend Planning (4 hours)

**Frontend Component Structure:**
```
frontend/src/
├─ components/
│  ├─ Auth/
│  │  ├─ LoginPage.jsx
│  │  ├─ SignupPage.jsx
│  │  └─ ProtectedRoute.jsx
│  ├─ Dashboard/
│  │  ├─ ProjectList.jsx
│  │  ├─ ProjectCard.jsx
│  │  └─ NewProjectModal.jsx
│  ├─ CompoundUpload/
│  │  ├─ CSVUploader.jsx
│  │  ├─ CompoundPreview.jsx
│  │  └─ ValidationChecker.jsx
│  ├─ QSARTraining/
│  │  ├─ ModelBuilder.jsx
│  │  ├─ FeatureSelector.jsx
│  │  ├─ TrainingProgress.jsx
│  │  └─ ModelStats.jsx
│  ├─ Predictions/
│  │  ├─ PredictionForm.jsx
│  │  ├─ BatchPrediction.jsx
│  │  └─ ResultsDisplay.jsx
│  └─ Common/
│     ├─ Navbar.jsx
│     ├─ Sidebar.jsx
│     └─ LoadingSpinner.jsx
├─ pages/
│  ├─ HomePage.jsx
│  ├─ DashboardPage.jsx
│  ├─ ProjectPage.jsx
│  └─ NotFoundPage.jsx
├─ services/
│  ├─ api.js
│  ├─ auth.js
│  └─ storage.js
├─ hooks/
│  ├─ useAuth.js
│  ├─ useProject.js
│  └─ useAPI.js
├─ context/
│  └─ AuthContext.jsx
└─ App.jsx
```

**Deliverables:**
- [ ] Frontend folder structure created
- [ ] Component list finalized
- [ ] UI wireframes sketched (can be simple)

---

## PHASE 2: BACKEND DEVELOPMENT (WEEKS 2-3)

### Week 2 Goals: FastAPI Core Setup & Authentication

**Hours: 40 hours total**

#### Day 1-2: FastAPI Scaffolding (8 hours)

```bash
# backend/requirements.txt
fastapi==0.104.1
uvicorn==0.24.0
python-dotenv==1.0.0
pydantic==2.5.0
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
python-jose==3.3.0
passlib==1.7.4
bcrypt==4.1.1
email-validator==2.1.0

# Chemistry & ML
transformers==4.35.2
torch==2.1.1
scikit-learn==1.3.2
rdkit==2023.09.0
pandas==2.1.3

# Task Scheduling
apscheduler==3.10.4
python-dateutil==2.8.2

# API & Async
aiofiles==23.2.1
httpx==0.25.2
```

**Main Backend Structure:**
```python
# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

load_dotenv()

# Import routers
from routers import auth, projects, compounds, qsar, predictions, jobs

# Lifespan context for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting AI.BIODOCKIFY backend...")
    # Initialize scheduler for background jobs
    yield
    # Shutdown
    print("Shutting down...")

app = FastAPI(
    title="AI.BIODOCKIFY API",
    description="Auto-QSAR & Molecular Docking Platform",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-app.vercel.app",
        "http://localhost:3000",
        os.getenv("FRONTEND_URL", "http://localhost:3000")
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(projects.router, prefix="/api/v1/projects", tags=["projects"])
app.include_router(compounds.router, prefix="/api/v1/compounds", tags=["compounds"])
app.include_router(qsar.router, prefix="/api/v1/qsar", tags=["qsar"])
app.include_router(predictions.router, prefix="/api/v1/predictions", tags=["predictions"])
app.include_router(jobs.router, prefix="/api/v1/jobs", tags=["jobs"])

@app.get("/health")
async def health():
    return {"status": "ok", "service": "ai.biodockify"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Database Connection:**
```python
# backend/database.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

DATABASE_URL = os.getenv("DATABASE_URL")
# Fix postgres:// to postgresql:// for SQLAlchemy 2.0+
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

**Deliverables:**
- [ ] FastAPI project initialized
- [ ] Requirements.txt with all dependencies
- [ ] Main app file with router imports
- [ ] Database connection configured
- [ ] CORS setup for Vercel frontend

#### Day 3-4: Authentication System (8 hours)

```python
# backend/models/user.py
from sqlalchemy import Column, String, DateTime
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime
from database import Base
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(255))
    hashed_password = Column(String(255), nullable=False)
    api_key = Column(String(255), unique=True, index=True)
    organization = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def verify_password(self, password: str) -> bool:
        return pwd_context.verify(password, self.hashed_password)
    
    def set_password(self, password: str):
        self.hashed_password = pwd_context.hash(password)

# backend/schemas/auth.py
from pydantic import BaseModel, EmailStr
from typing import Optional

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    organization: Optional[str]
    
    class Config:
        from_attributes = True

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

# backend/routers/auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import timedelta
from models.user import User
from schemas.auth import UserCreate, UserLogin, TokenResponse, UserResponse
from database import get_db
from security import create_access_token, verify_token, get_current_user

router = APIRouter()

@router.post("/register", response_model=TokenResponse)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    # Check if user exists
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create new user
    user = User(email=user_data.email, name=user_data.name)
    user.set_password(user_data.password)
    
    db.add(user)
    db.commit()
    db.refresh(user)
    
    # Generate token
    access_token = create_access_token(data={"sub": str(user.id)})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": UserResponse.from_orm(user)
    }

@router.post("/login", response_model=TokenResponse)
async def login(credentials: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == credentials.email).first()
    
    if not user or not user.verify_password(credentials.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = create_access_token(data={"sub": str(user.id)})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": UserResponse.from_orm(user)
    }

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    return UserResponse.from_orm(current_user)
```

**Security Module:**
```python
# backend/security.py
from datetime import datetime, timedelta
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthCredentials
from sqlalchemy.orm import Session
import os
from models.user import User
from database import get_db

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30 * 24 * 60  # 30 days

security = HTTPBearer()

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(credentials: HTTPAuthCredentials = Depends(security), db: Session = Depends(get_db)):
    token = credentials.credentials
    user_id = verify_token(token)
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user
```

**Deliverables:**
- [ ] User model created with password hashing
- [ ] Authentication schemas (register, login)
- [ ] Auth router with register/login endpoints
- [ ] Security module with JWT token handling
- [ ] Password verification working

#### Day 5: Projects & Compounds Routers (8 hours)

```python
# backend/models/project.py
from sqlalchemy import Column, String, Text, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime
from database import Base

class Project(Base):
    __tablename__ = "projects"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    name = Column(String(255), nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    compounds = relationship("Compound", back_populates="project", cascade="all, delete-orphan")
    qsar_models = relationship("QSARModel", back_populates="project", cascade="all, delete-orphan")

# backend/models/compound.py
from sqlalchemy import Column, String, ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime
from database import Base

class Compound(Base):
    __tablename__ = "compounds"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"))
    smiles = Column(String(1000), nullable=False)
    inchi_key = Column(String(255))
    name = Column(String(255))
    source = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    project = relationship("Project", back_populates="compounds")

# backend/routers/projects.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from models.user import User
from models.project import Project
from schemas.project import ProjectCreate, ProjectUpdate, ProjectResponse
from database import get_db
from security import get_current_user

router = APIRouter()

@router.post("", response_model=ProjectResponse)
async def create_project(
    project_data: ProjectCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    project = Project(
        user_id=current_user.id,
        name=project_data.name,
        description=project_data.description
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    return ProjectResponse.from_orm(project)

@router.get("", response_model=list[ProjectResponse])
async def list_projects(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    projects = db.query(Project).filter(Project.user_id == current_user.id).all()
    return [ProjectResponse.from_orm(p) for p in projects]

@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    return ProjectResponse.from_orm(project)

@router.put("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: str,
    project_data: ProjectUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if project_data.name:
        project.name = project_data.name
    if project_data.description is not None:
        project.description = project_data.description
    
    db.commit()
    db.refresh(project)
    return ProjectResponse.from_orm(project)

@router.delete("/{project_id}")
async def delete_project(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    db.delete(project)
    db.commit()
    return {"message": "Project deleted"}
```

**Deliverables:**
- [ ] Project model created with relationships
- [ ] Compound model created
- [ ] Projects router with CRUD operations
- [ ] Proper authorization checks (user can only access own projects)

### Week 3 Goals: QSAR Training Backend

**Hours: 40 hours total**

#### Day 1-3: ChemBERTa Integration & QSAR Training (16 hours)

```python
# backend/models/qsar_model.py
from sqlalchemy import Column, String, Float, Integer, Text, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime
from database import Base

class QSARModel(Base):
    __tablename__ = "qsar_models"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"))
    model_name = Column(String(255), nullable=False)
    model_type = Column(String(50))  # 'regression', 'classification'
    r2_score = Column(Float)
    rmse = Column(Float)
    mae = Column(Float)
    features_used = Column(Text)  # JSON
    target_variable = Column(String(255))
    version = Column(Integer, default=1)
    model_path = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    project = relationship("Project", back_populates="qsar_models")

# backend/services/chemberta_service.py
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import os
from typing import List, Tuple

class ChemBERTaService:
    def __init__(self):
        self.model_name = "DeepChem/ChemBERTa-77M-MLM"
        print(f"Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")
    
    def get_embeddings(self, smiles_list: List[str]) -> np.ndarray:
        """
        Get molecular embeddings from SMILES strings
        Returns: (n_compounds, 384) numpy array
        """
        embeddings = []
        
        with torch.no_grad():
            for smiles in smiles_list:
                try:
                    tokens = self.tokenizer(
                        smiles,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    ).to(self.device)
                    
                    outputs = self.model(**tokens)
                    # Use [CLS] token embedding (first token)
                    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    embeddings.append(embedding[0])
                except Exception as e:
                    print(f"Error processing SMILES {smiles}: {e}")
                    embeddings.append(np.zeros(384))
        
        return np.array(embeddings)

# backend/services/qsar_trainer.py
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime

class QSARTrainer:
    def __init__(self, chemberta_service):
        self.chemberta = chemberta_service
        self.scaler = StandardScaler()
    
    def train_model(
        self,
        smiles_list: list,
        target_values: list,
        model_name: str,
        model_type: str = "regression",
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """
        Train QSAR model using ChemBERTa embeddings
        """
        print(f"Training QSAR model: {model_name}")
        
        # Step 1: Get embeddings
        print("Generating ChemBERTa embeddings...")
        X = self.chemberta.get_embeddings(smiles_list)
        y = np.array(target_values)
        
        # Step 2: Scale features
        print("Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Step 3: Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )
        
        # Step 4: Train model
        print(f"Training {model_type} model...")
        if model_type == "regression":
            model = RandomForestRegressor(n_estimators=100, random_state=random_state)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        
        model.fit(X_train, y_train)
        
        # Step 5: Evaluate
        print("Evaluating model...")
        y_pred = model.predict(X_test)
        
        metrics = {
            "r2_score": float(r2_score(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae": float(mean_absolute_error(y_test, y_pred))
        }
        
        print(f"Model performance - R²: {metrics['r2_score']:.4f}, RMSE: {metrics['rmse']:.4f}")
        
        return {
            "model": model,
            "scaler": self.scaler,
            "metrics": metrics,
            "features_used": [f"ChemBERTa_{i}" for i in range(X.shape[1])],
            "n_compounds": len(smiles_list)
        }
    
    def predict(self, model, scaler, smiles_list: list) -> list:
        """Make predictions on new compounds"""
        X = self.chemberta.get_embeddings(smiles_list)
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        return predictions.tolist()

# backend/routers/qsar.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from models.user import User
from models.project import Project
from models.qsar_model import QSARModel
from database import get_db
from security import get_current_user
from services.chemberta_service import ChemBERTaService
from services.qsar_trainer import QSARTrainer
from schemas.qsar import QSARTrainRequest, QSARModelResponse, PredictionRequest
import pandas as pd
import json
import os
from io import StringIO

router = APIRouter()

# Initialize services
chemberta_service = ChemBERTaService()
qsar_trainer = QSARTrainer(chemberta_service)

@router.post("/{project_id}/finetune")
async def finetune_qsar(
    project_id: str,
    file: UploadFile = File(...),
    smiles_column: str = "SMILES",
    target_column: str = "Activity",
    model_name: str = "QSAR_Model_v1",
    model_type: str = "regression",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Fine-tune QSAR model on user's data
    Expects CSV with SMILES and target values
    """
    try:
        # Verify project access
        project = db.query(Project).filter(
            Project.id == project_id,
            Project.user_id == current_user.id
        ).first()
        
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Read CSV
        content = await file.read()
        df = pd.read_csv(StringIO(content.decode('utf-8')))
        
        if smiles_column not in df.columns or target_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"CSV must contain '{smiles_column}' and '{target_column}' columns"
            )
        
        smiles_list = df[smiles_column].dropna().tolist()
        target_values = df[target_column].dropna().tolist()
        
        if len(smiles_list) < 10:
            raise HTTPException(
                status_code=400,
                detail="Need at least 10 valid compounds to train"
            )
        
        # Train model
        result = qsar_trainer.train_model(
            smiles_list=smiles_list,
            target_values=target_values,
            model_name=model_name,
            model_type=model_type
        )
        
        # Save to database
        qsar_model = QSARModel(
            project_id=project_id,
            model_name=model_name,
            model_type=model_type,
            r2_score=result["metrics"]["r2_score"],
            rmse=result["metrics"]["rmse"],
            mae=result["metrics"]["mae"],
            features_used=json.dumps(result["features_used"]),
            target_variable=target_column,
            version=1
        )
        
        db.add(qsar_model)
        db.commit()
        db.refresh(qsar_model)
        
        # Save model files locally (TODO: move to S3)
        model_dir = f"models/{project_id}/{qsar_model.id}"
        os.makedirs(model_dir, exist_ok=True)
        
        import pickle
        with open(f"{model_dir}/model.pkl", "wb") as f:
            pickle.dump(result["model"], f)
        with open(f"{model_dir}/scaler.pkl", "wb") as f:
            pickle.dump(result["scaler"], f)
        
        return {
            "model_id": str(qsar_model.id),
            "model_name": model_name,
            "metrics": result["metrics"],
            "n_compounds_used": len(smiles_list),
            "message": "Model trained successfully"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{project_id}/models")
async def list_models(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    models = db.query(QSARModel).filter(QSARModel.project_id == project_id).all()
    return [QSARModelResponse.from_orm(m) for m in models]

@router.post("/{project_id}/predict")
async def predict(
    project_id: str,
    request: PredictionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Predict on single or batch SMILES"""
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Get model
    model_obj = db.query(QSARModel).filter(QSARModel.id == request.model_id).first()
    
    if not model_obj:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Load model and scaler
    import pickle
    model_dir = f"models/{project_id}/{model_obj.id}"
    
    with open(f"{model_dir}/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(f"{model_dir}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    # Predict
    predictions = qsar_trainer.predict(model, scaler, request.smiles_list)
    
    return {
        "model_id": str(model_obj.id),
        "predictions": predictions,
        "n_compounds": len(request.smiles_list)
    }
```

**Deliverables:**
- [ ] ChemBERTa service for embeddings
- [ ] QSAR trainer with Random Forest implementation
- [ ] QSAR router with fine-tune and predict endpoints
- [ ] Model saving/loading functionality
- [ ] CSV upload and validation

#### Day 4-5: Background Jobs & Testing (8 hours)

```python
# backend/services/job_scheduler.py
from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy.orm import Session
from models.training_job import TrainingJob
from database import SessionLocal
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JobScheduler:
    def __init__(self):
        self.scheduler = BackgroundScheduler()
    
    def start(self):
        self.scheduler.start()
        logger.info("Job scheduler started")
    
    def check_training_jobs(self):
        """Check for pending training jobs and process them"""
        db = SessionLocal()
        try:
            pending_jobs = db.query(TrainingJob).filter(
                TrainingJob.status == 'queued'
            ).all()
            
            for job in pending_jobs:
                logger.info(f"Processing training job {job.id}")
                # TODO: Process job (move to separate service)
        finally:
            db.close()

job_scheduler = JobScheduler()
```

**Deliverables:**
- [ ] Background job scheduler setup
- [ ] Job status tracking
- [ ] Error handling and logging

---

## PHASE 3: FRONTEND DEVELOPMENT (WEEKS 4-5)

### Week 4 Goals: React Setup & Core Components

**Hours: 40 hours**

#### Day 1-2: React Scaffolding (8 hours)

```bash
# Create React app with Vercel template
npx create-react-app frontend
cd frontend

# Install dependencies
npm install axios react-router-dom zustand

# Create folder structure
mkdir -p src/{components,pages,services,hooks,context,utils}

# .env.local (don't commit)
REACT_APP_API_URL=http://localhost:8000
REACT_APP_API_URL_PROD=https://username-api.hf.space
```

#### Day 3-4: Authentication & Dashboard (16 hours)

```javascript
// frontend/src/context/AuthContext.jsx
import React, { createContext, useState, useEffect } from 'react';

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(localStorage.getItem('token'));
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (token) {
      // Verify token by calling /me endpoint
      validateToken();
    } else {
      setLoading(false);
    }
  }, []);

  const validateToken = async () => {
    try {
      const response = await fetch(
        `${process.env.REACT_APP_API_URL}/api/v1/auth/me`,
        {
          headers: { Authorization: `Bearer ${token}` }
        }
      );
      if (response.ok) {
        const userData = await response.json();
        setUser(userData);
      } else {
        localStorage.removeItem('token');
        setToken(null);
      }
    } finally {
      setLoading(false);
    }
  };

  const login = async (email, password) => {
    const response = await fetch(
      `${process.env.REACT_APP_API_URL}/api/v1/auth/login`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      }
    );
    if (!response.ok) throw new Error('Login failed');
    const data = await response.json();
    setToken(data.access_token);
    setUser(data.user);
    localStorage.setItem('token', data.access_token);
    return data;
  };

  const register = async (email, password, name) => {
    const response = await fetch(
      `${process.env.REACT_APP_API_URL}/api/v1/auth/register`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password, name })
      }
    );
    if (!response.ok) throw new Error('Registration failed');
    const data = await response.json();
    setToken(data.access_token);
    setUser(data.user);
    localStorage.setItem('token', data.access_token);
    return data;
  };

  const logout = () => {
    setToken(null);
    setUser(null);
    localStorage.removeItem('token');
  };

  return (
    <AuthContext.Provider value={{ user, token, loading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = React.useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
};

// frontend/src/pages/LoginPage.jsx
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function LoginPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      await login(email, password);
      navigate('/dashboard');
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div style={styles.container}>
      <h1>AI.BIODOCKIFY</h1>
      <h2>Login</h2>
      {error && <p style={styles.error}>{error}</p>}
      <form onSubmit={handleSubmit} style={styles.form}>
        <input
          type="email"
          placeholder="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
          style={styles.input}
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
          style={styles.input}
        />
        <button type="submit" style={styles.button}>Login</button>
      </form>
      <p>
        Don't have an account? <a href="/register">Register</a>
      </p>
    </div>
  );
}

const styles = {
  container: {
    maxWidth: '400px',
    margin: '100px auto',
    padding: '20px',
    textAlign: 'center'
  },
  form: {
    display: 'flex',
    flexDirection: 'column'
  },
  input: {
    padding: '10px',
    margin: '10px 0',
    border: '1px solid #ccc',
    borderRadius: '5px'
  },
  button: {
    padding: '10px',
    background: '#0070f3',
    color: 'white',
    border: 'none',
    borderRadius: '5px',
    cursor: 'pointer'
  },
  error: {
    color: 'red',
    marginBottom: '10px'
  }
};

// frontend/src/pages/DashboardPage.jsx
import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';

export default function DashboardPage() {
  const { user, token } = useAuth();
  const [projects, setProjects] = useState([]);
  const [newProjectName, setNewProjectName] = useState('');

  useEffect(() => {
    fetchProjects();
  }, []);

  const fetchProjects = async () => {
    const response = await fetch(
      `${process.env.REACT_APP_API_URL}/api/v1/projects`,
      {
        headers: { Authorization: `Bearer ${token}` }
      }
    );
    if (response.ok) {
      const data = await response.json();
      setProjects(data);
    }
  };

  const handleCreateProject = async (e) => {
    e.preventDefault();
    const response = await fetch(
      `${process.env.REACT_APP_API_URL}/api/v1/projects`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({ name: newProjectName, description: '' })
      }
    );
    if (response.ok) {
      setNewProjectName('');
      fetchProjects();
    }
  };

  return (
    <div style={styles.container}>
      <h1>Dashboard - {user?.name}</h1>
      <form onSubmit={handleCreateProject} style={styles.form}>
        <input
          type="text"
          placeholder="Project name"
          value={newProjectName}
          onChange={(e) => setNewProjectName(e.target.value)}
          style={styles.input}
        />
        <button type="submit" style={styles.button}>Create Project</button>
      </form>
      <h2>Your Projects</h2>
      {projects.length === 0 ? (
        <p>No projects yet. Create one to get started!</p>
      ) : (
        <div style={styles.grid}>
          {projects.map((project) => (
            <div key={project.id} style={styles.card}>
              <h3>{project.name}</h3>
              <p>{project.description || 'No description'}</p>
              <button style={styles.button}>Open</button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

const styles = {
  container: { maxWidth: '1000px', margin: '0 auto', padding: '20px' },
  form: { display: 'flex', gap: '10px', marginBottom: '20px' },
  input: { flex: 1, padding: '10px', border: '1px solid #ccc', borderRadius: '5px' },
  button: { padding: '10px 20px', background: '#0070f3', color: 'white', border: 'none', borderRadius: '5px' },
  grid: { display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))', gap: '20px' },
  card: { border: '1px solid #eee', padding: '20px', borderRadius: '5px' }
};
```

#### Day 5: Routing & Basic Pages (8 hours)

```javascript
// frontend/src/App.jsx
import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import DashboardPage from './pages/DashboardPage';
import ProjectPage from './pages/ProjectPage';

function ProtectedRoute({ children }) {
  const { token, loading } = useAuth();
  
  if (loading) return <div>Loading...</div>;
  if (!token) return <Navigate to="/login" />;
  return children;
}

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />
          <Route path="/dashboard" element={
            <ProtectedRoute>
              <DashboardPage />
            </ProtectedRoute>
          } />
          <Route path="/projects/:projectId" element={
            <ProtectedRoute>
              <ProjectPage />
            </ProtectedRoute>
          } />
          <Route path="/" element={<Navigate to="/dashboard" />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}
```

**Deliverables:**
- [ ] React project initialized
- [ ] Authentication context created
- [ ] Login/Register pages functional
- [ ] Dashboard with project list
- [ ] Routing setup with protected routes

### Week 5 Goals: QSAR Training UI & File Upload

**Hours: 40 hours**

#### Day 1-3: File Upload & QSAR Training UI (16 hours)

```javascript
// frontend/src/pages/ProjectPage.jsx
import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import CSVUploader from '../components/CSVUploader';
import QSARTrainingForm from '../components/QSARTrainingForm';
import ModelsList from '../components/ModelsList';

export default function ProjectPage() {
  const { projectId } = useParams();
  const { token } = useAuth();
  const [project, setProject] = useState(null);
  const [compounds, setCompounds] = useState([]);
  const [models, setModels] = useState([]);
  const [activeTab, setActiveTab] = useState('upload');

  useEffect(() => {
    fetchProject();
    fetchModels();
  }, [projectId]);

  const fetchProject = async () => {
    const response = await fetch(
      `${process.env.REACT_APP_API_URL}/api/v1/projects/${projectId}`,
      { headers: { Authorization: `Bearer ${token}` } }
    );
    if (response.ok) {
      setProject(await response.json());
    }
  };

  const fetchModels = async () => {
    const response = await fetch(
      `${process.env.REACT_APP_API_URL}/api/v1/qsar/${projectId}/models`,
      { headers: { Authorization: `Bearer ${token}` } }
    );
    if (response.ok) {
      setModels(await response.json());
    }
  };

  if (!project) return <div>Loading...</div>;

  return (
    <div style={styles.container}>
      <h1>{project.name}</h1>
      
      <div style={styles.tabs}>
        <button
          style={{...styles.tab, ...( activeTab === 'upload' ? styles.activeTab : {})}}
          onClick={() => setActiveTab('upload')}
        >
          Upload Data
        </button>
        <button
          style={{...styles.tab, ...( activeTab === 'train' ? styles.activeTab : {})}}
          onClick={() => setActiveTab('train')}
        >
          Train Model
        </button>
        <button
          style={{...styles.tab, ...( activeTab === 'models' ? styles.activeTab : {})}}
          onClick={() => setActiveTab('models')}
        >
          My Models ({models.length})
        </button>
      </div>

      {activeTab === 'upload' && (
        <CSVUploader projectId={projectId} token={token} onUploadSuccess={fetchProject} />
      )}
      
      {activeTab === 'train' && (
        <QSARTrainingForm projectId={projectId} token={token} onTrainSuccess={fetchModels} />
      )}
      
      {activeTab === 'models' && (
        <ModelsList models={models} projectId={projectId} />
      )}
    </div>
  );
}

const styles = {
  container: { maxWidth: '1000px', margin: '0 auto', padding: '20px' },
  tabs: { display: 'flex', gap: '10px', marginBottom: '20px', borderBottom: '1px solid #eee' },
  tab: { padding: '10px 20px', background: 'none', border: 'none', cursor: 'pointer', borderBottom: '2px solid transparent' },
  activeTab: { borderBottomColor: '#0070f3', color: '#0070f3' }
};

// frontend/src/components/CSVUploader.jsx
import React, { useState } from 'react';

export default function CSVUploader({ projectId, token, onUploadSuccess }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [preview, setPreview] = useState(null);

  const handleFileSelect = async (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;
    
    setFile(selectedFile);
    
    // Read and preview first 5 rows
    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target.result;
      const lines = text.split('\n').slice(0, 6);
      setPreview(lines);
    };
    reader.readAsText(selectedFile);
  };

  const handleUpload = async () => {
    if (!file) return;
    
    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(
        `${process.env.REACT_APP_API_URL}/api/v1/compounds/${projectId}/upload`,
        {
          method: 'POST',
          headers: { Authorization: `Bearer ${token}` },
          body: formData
        }
      );
      
      if (response.ok) {
        const result = await response.json();
        alert(`Uploaded ${result.n_compounds} compounds!`);
        setFile(null);
        setPreview(null);
        onUploadSuccess();
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      <h2>Upload Compounds</h2>
      <p>Upload a CSV file with SMILES strings and target values</p>
      
      <input
        type="file"
        accept=".csv"
        onChange={handleFileSelect}
        style={styles.input}
      />
      
      {preview && (
        <div style={styles.preview}>
          <h3>Preview:</h3>
          {preview.map((line, i) => (
            <code key={i}>{line}</code>
          ))}
        </div>
      )}
      
      <button
        onClick={handleUpload}
        disabled={!file || loading}
        style={styles.button}
      >
        {loading ? 'Uploading...' : 'Upload'}
      </button>
    </div>
  );
}

const styles = {
  container: { padding: '20px', border: '1px solid #eee', borderRadius: '5px' },
  input: { display: 'block', margin: '10px 0' },
  preview: { background: '#f5f5f5', padding: '10px', margin: '10px 0', borderRadius: '5px', overflow: 'auto' },
  button: { padding: '10px 20px', background: '#0070f3', color: 'white', border: 'none', borderRadius: '5px' }
};

// frontend/src/components/QSARTrainingForm.jsx
import React, { useState } from 'react';

export default function QSARTrainingForm({ projectId, token, onTrainSuccess }) {
  const [formData, setFormData] = useState({
    modelName: 'QSAR_Model_v1',
    modelType: 'regression',
    smilesColumn: 'SMILES',
    targetColumn: 'Activity'
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleTrain = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await fetch(
        `${process.env.REACT_APP_API_URL}/api/v1/qsar/${projectId}/finetune`,
        {
          method: 'POST',
          headers: { Authorization: `Bearer ${token}` },
          body: JSON.stringify(formData)
        }
      );

      if (response.ok) {
        const result = await response.json();
        setResult(result);
        onTrainSuccess();
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      <h2>Train QSAR Model</h2>
      <form onSubmit={handleTrain}>
        <div style={styles.formGroup}>
          <label>Model Name</label>
          <input
            type="text"
            name="modelName"
            value={formData.modelName}
            onChange={handleChange}
            style={styles.input}
          />
        </div>

        <div style={styles.formGroup}>
          <label>Model Type</label>
          <select
            name="modelType"
            value={formData.modelType}
            onChange={handleChange}
            style={styles.input}
          >
            <option>regression</option>
            <option>classification</option>
          </select>
        </div>

        <div style={styles.formGroup}>
          <label>SMILES Column Name</label>
          <input
            type="text"
            name="smilesColumn"
            value={formData.smilesColumn}
            onChange={handleChange}
            style={styles.input}
          />
        </div>

        <div style={styles.formGroup}>
          <label>Target Column Name</label>
          <input
            type="text"
            name="targetColumn"
            value={formData.targetColumn}
            onChange={handleChange}
            style={styles.input}
          />
        </div>

        <button type="submit" disabled={loading} style={styles.button}>
          {loading ? 'Training...' : 'Train Model'}
        </button>
      </form>

      {result && (
        <div style={styles.result}>
          <h3>Training Complete!</h3>
          <p>R² Score: {result.metrics.r2_score.toFixed(4)}</p>
          <p>RMSE: {result.metrics.rmse.toFixed(4)}</p>
          <p>MAE: {result.metrics.mae.toFixed(4)}</p>
          <p>Compounds Used: {result.n_compounds_used}</p>
        </div>
      )}
    </div>
  );
}

const styles = {
  container: { padding: '20px', border: '1px solid #eee', borderRadius: '5px' },
  formGroup: { marginBottom: '15px' },
  input: { width: '100%', padding: '8px', border: '1px solid #ccc', borderRadius: '3px' },
  button: { padding: '10px 20px', background: '#0070f3', color: 'white', border: 'none', borderRadius: '5px' },
  result: { marginTop: '20px', padding: '15px', background: '#e8f5e9', borderRadius: '5px' }
};

// frontend/src/components/ModelsList.jsx
import React from 'react';

export default function ModelsList({ models, projectId }) {
  return (
    <div>
      <h2>Your QSAR Models</h2>
      {models.length === 0 ? (
        <p>No models trained yet.</p>
      ) : (
        <div style={styles.grid}>
          {models.map((model) => (
            <div key={model.id} style={styles.card}>
              <h3>{model.model_name}</h3>
              <p>Type: {model.model_type}</p>
              <p>R² Score: {model.r2_score?.toFixed(4) || 'N/A'}</p>
              <p>RMSE: {model.rmse?.toFixed(4) || 'N/A'}</p>
              <button style={styles.button}>Use Model</button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

const styles = {
  grid: { display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))', gap: '20px' },
  card: { border: '1px solid #eee', padding: '15px', borderRadius: '5px' },
  button: { padding: '8px 16px', background: '#0070f3', color: 'white', border: 'none', borderRadius: '3px' }
};
```

**Deliverables:**
- [ ] Project detail page created
- [ ] CSV file uploader component
- [ ] QSAR training form
- [ ] Models list display
- [ ] File preview functionality

#### Day 4-5: Predictions & Polish (8 hours)

```javascript
// frontend/src/components/PredictionForm.jsx
import React, { useState } from 'react';

export default function PredictionForm({ modelId, projectId, token }) {
  const [smilesInput, setSmilesInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [predictions, setPredictions] = useState(null);

  const handlePredict = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const smilesList = smilesInput.split('\n').filter(s => s.trim());
      const response = await fetch(
        `${process.env.REACT_APP_API_URL}/api/v1/qsar/${projectId}/predict`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}`
          },
          body: JSON.stringify({ model_id: modelId, smiles_list: smilesList })
        }
      );

      if (response.ok) {
        const result = await response.json();
        setPredictions(result);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      <h2>Make Predictions</h2>
      <form onSubmit={handlePredict}>
        <textarea
          placeholder="Enter SMILES strings (one per line)"
          value={smilesInput}
          onChange={(e) => setSmilesInput(e.target.value)}
          style={styles.textarea}
        />
        <button type="submit" disabled={loading} style={styles.button}>
          {loading ? 'Predicting...' : 'Predict'}
        </button>
      </form>

      {predictions && (
        <div style={styles.results}>
          <h3>Predictions</h3>
          {predictions.predictions.map((pred, i) => (
            <div key={i} style={styles.predictionRow}>
              <span>SMILES {i + 1}: </span>
              <strong>{pred.toFixed(4)}</strong>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

const styles = {
  container: { padding: '20px', border: '1px solid #eee', borderRadius: '5px' },
  textarea: { width: '100%', height: '150px', padding: '10px', border: '1px solid #ccc', borderRadius: '3px' },
  button: { padding: '10px 20px', background: '#0070f3', color: 'white', border: 'none', borderRadius: '5px', marginTop: '10px' },
  results: { marginTop: '20px', padding: '15px', background: '#f0f7ff', borderRadius: '5px' },
  predictionRow: { padding: '8px', borderBottom: '1px solid #eee' }
};
```

**Deliverables:**
- [ ] Prediction component created
- [ ] Integration with models list
- [ ] Results display with formatting
- [ ] Error handling

---

## PHASE 4: DEPLOYMENT & TESTING (WEEKS 6-8)

### Week 6: Backend Deployment (HF Spaces)

```bash
# Create HF Space
# 1. Go to huggingface.co/spaces
# 2. Create new Space: "biodockify-api"
# 3. Select Docker SDK
# 4. Create local git repo for HF:
cd backend
git init
git remote add space https://huggingface.co/spaces/yourname/biodockify-api

# Create Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

# Create .env for HF (add to Dockerfile)
# ENV DATABASE_URL=your_supabase_url
# ENV SECRET_KEY=your_secret_key

git add .
git commit -m "Deploy backend to HF Spaces"
git push space main
```

**Deliverables:**
- [ ] Backend deployed on HF Spaces
- [ ] Health endpoint responding
- [ ] Database connection working
- [ ] CORS configured for Vercel
- [ ] Environment variables set

### Week 7: Frontend Deployment (Vercel)

```bash
# In frontend directory
# 1. Push to GitHub
git push origin main

# 2. Go to vercel.com
# 3. Import GitHub repository
# 4. Set environment variable:
REACT_APP_API_URL=https://yourname-biodockify-api.hf.space

# 5. Deploy!
# Vercel auto-deploys on every push to main
```

**Deliverables:**
- [ ] Frontend deployed on Vercel
- [ ] Environment variables configured
- [ ] HTTPS working
- [ ] API calls to backend successful
- [ ] Authentication working end-to-end

### Week 8: Testing & Polish

**Integration Testing:**
- [ ] User registration flow
- [ ] Login/logout
- [ ] Project creation
- [ ] CSV upload (small file)
- [ ] QSAR model training (5-10 minutes)
- [ ] Predictions on new compounds
- [ ] Model list and stats display

**Performance Testing:**
- [ ] Frontend load time < 3 seconds
- [ ] API response time < 2 seconds
- [ ] QSAR training with 50 compounds
- [ ] Batch prediction (100 compounds)

**Bug Fixes & Polish:**
- [ ] Error messages display correctly
- [ ] Loading states show
- [ ] Mobile responsive (basic)
- [ ] Console has no errors
- [ ] Documentation updated

---

## TIMELINE SUMMARY

```
WEEK 1: Planning & Setup (20 hrs)
├─ Accounts, infrastructure, GitHub setup
├─ Database schema
└─ API specification

WEEK 2-3: Backend (80 hrs)
├─ FastAPI core, authentication
├─ Projects/compounds routers
├─ ChemBERTa integration
├─ QSAR training pipeline
└─ Job scheduling

WEEK 4-5: Frontend (80 hrs)
├─ React scaffolding
├─ Auth pages
├─ Dashboard & projects
├─ CSV upload UI
├─ QSAR training UI
└─ Predictions

WEEK 6: Backend Deployment (20 hrs)
├─ Docker setup
├─ HF Spaces deployment
├─ Environment config
└─ Testing

WEEK 7: Frontend Deployment (10 hrs)
├─ Vercel setup
├─ Environment config
└─ Production testing

WEEK 8: Testing & Polish (20 hrs)
├─ Integration testing
├─ Performance optimization
├─ Bug fixes
└─ Final documentation

TOTAL: 250 hours = ~6 weeks full-time
```

---

## MILESTONE CHECKPOINTS

**End of Week 1:** Infrastructure ready, database designed, API spec finalized
**End of Week 3:** Full backend with QSAR training working locally
**End of Week 5:** Full frontend with all pages, connects to backend
**End of Week 6:** Backend deployed and working on HF Spaces
**End of Week 7:** Frontend deployed on Vercel, end-to-end working
**End of Week 8:** Production-ready MVP launched

---

## SUCCESS CRITERIA

- [ ] Users can register & login
- [ ] Users can create projects
- [ ] Users can upload CSV files
- [ ] Users can train QSAR models (5-10 min per model)
- [ ] Users can make predictions
- [ ] All endpoints have < 2 second response time
- [ ] Zero monthly cost
- [ ] Can handle 50 concurrent users
- [ ] Documentation complete
- [ ] Code on GitHub with commits
- [ ] Both apps deployed (Vercel + HF Spaces)

---

## POST-MVP ROADMAP

**Phase 2 (Weeks 9-12):**
- Molecular docking integration (GNINA)
- Batch job processing
- Model comparison/evaluation tools
- Advanced visualization

**Phase 3 (Weeks 13-16):**
- Custom descriptor support
- Ensemble models
- REST API documentation (OpenAPI)
- User roles (free/premium)

**Phase 4+ (Weeks 17+):**
- Web 3D molecular viewer
- Paid tier with GPU access
- Academic partnerships
- Published paper on platform

---

## COST ANALYSIS (8 WEEKS)

```
Accounts: $0 (all free)
Hosting: $0 (Vercel + HF Spaces free)
Database: $0 (Supabase free)
ML: $0 (self-hosted ChemBERTa)
Tools: $0 (all open source)
Development Cost: Your time (250 hours ÷ 8 weeks)

Total Monthly Ongoing: $0 ✅
```

**YOU CAN LAUNCH A PRODUCTION-READY PLATFORM FOR ZERO COST IN 8 WEEKS!** 🚀

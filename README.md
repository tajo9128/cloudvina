---
title: BioDockify AI
emoji: 🧬
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# BioDockify

**Democratizing Molecular Docking for Students and Researchers**

BioDockify is a web-based SaaS platform that makes AutoDock Vina accessible to anyone with a browser. No Linux, no command line, no expensive hardware required.

## 🎯 Problem We Solve

- **Technical Barrier**: AutoDock Vina requires Linux CLI expertise
- **Hardware Limitation**: Students' laptops can't handle intensive docking simulations
- **Cost Barrier**: Commercial tools cost $10k+ per license
- **Time Waste**: Thesis students spend weeks learning tools instead of doing research

## 💡 Our Solution

1. **Upload** your PDB receptor and ligand files (drag & drop)
2. **Click** "Start Docking"
3. **View** results in 3D (interactive molecule viewer)
4. **Download** output files for your thesis

**Pricing**: Pay-as-you-go ($0.10 per dock) or Student Plan ($9/mo for 50 docks)

## 🚀 Current Status: Phase 1 (Week 1-2)

✅ **Completed:**
- Docker container with AutoDock Vina
- S3 integration for file storage
- Python runner script
- AWS setup documentation

🔄 **In Progress:**
- Local Docker testing
- AWS Batch configuration

📋 **Next:**
- FastAPI backend (Phase 2)
- React frontend (Phase 3)
- Beta launch (Phase 4)

## 📁 Project Structure

```
BioDockify/
├── docker/           # AutoDock Vina container (Phase 1) ✅
├── backend/          # FastAPI server (Phase 2) 🔄
├── frontend/         # React web app (Phase 3) 📋
└── docs/            # Master plan & guides
```

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for details.

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | React + Vite + NGL Viewer |
| Backend | Python FastAPI |
| Database | Supabase (PostgreSQL) |
| Storage | AWS S3 |
| Compute | AWS Batch + Spot Instances |
| Container | Docker (AutoDock Vina) |

## 📖 Documentation

- **[Master Plan](docs/BioDockify_master_plan.md)** - Complete 7-week development blueprint
- **[Docker README](docker/README.md)** - How to build and run the container
- **[AWS Setup Guide](AWS_SETUP.md)** - Step-by-step AWS Free Tier configuration

## 🧪 Quick Start (Local Testing)

```bash
# 1. Build Docker image
cd docker
docker build -t BioDockify:latest .

# 2. Prepare test data
mkdir test_data
# (Add your receptor.pdb and ligand.sdf files)

# 3. Run docking locally
docker run --rm \
  -v $(pwd)/test_data:/app/work \
  -e JOB_ID=test-001 \
  -e RECEPTOR_S3_KEY=receptor.pdb \
  -e LIGAND_S3_KEY=ligand.sdf \
  BioDockify:latest
```

## 💰 Cost Structure

**Development (Free):**
- AWS Free Tier: 750 hrs/month t2.micro
- Supabase Free: 500MB database
- Vercel Free: Unlimited hosting
- **Total: $0.00/month**

**Production (1,000 jobs/month):**
- Compute: $10.00 (Spot Instances)
- Storage: $0.50 (S3)
- Database: $0.00 (Supabase Free)
- **Total: ~$11.50/month**

**Break-even:** 115 paid docks at $0.10 each

## 🎓 Target Audience

- M.Pharm & PhD students (thesis projects)
- Academic researchers (drug discovery)
- Small biotech startups (early-stage screening)
- Education (teaching molecular docking)

**Primary Market:** India, Southeast Asia, developing countries

## 📞 Support

- **Issues**: Use GitHub Issues for bugs
- **Questions**: See AWS_SETUP.md troubleshooting
- **Feature Requests**: Open a discussion

## 📜 License

This project uses AutoDock Vina (Apache 2.0 license). Commercial use permitted with attribution.

## 🗺️ Roadmap

- [x] **Phase 1** - Docker container (Week 1-2)
- [ ] **Phase 2** - FastAPI backend (Week 3-4)
- [ ] **Phase 3** - React frontend (Week 5-6)
- [ ] **Phase 4** - Beta launch (Week 7)
- [ ] **Phase 5** - Virtual screening (Q1 2026)
- [ ] **Phase 6** - AI binding prediction (Q2 2026)

---

**Built with ❤️ to make science more accessible**

---
# AI.BioDockify Backend Details

This is the backend service for the AI.BioDockify platform, running on Hugging Face Spaces.
It provides:
- QSAR Model Training (ChemBERTa)
- Toxicity Prediction
- Model Management

## API Documentation
Once running, visit `/docs` for Swagger UI.

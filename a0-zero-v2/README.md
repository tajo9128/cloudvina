# Agent Zero v2 IDE Improvements

## 📋 Overview
This directory contains all IDE-grade improvements for Agent Zero v2, transforming it from an AI assistant into a fully autonomous IDE-grade development system.

**Purpose**: Persistent backup of all improvements in GitHub. If Docker Desktop is deleted and a new Agent Zero instance is created, simply copy this folder to `/a0/lib/` to restore all capabilities.

---

## 🏗️ Architecture

### Phase 1: Foundation (Complete)
1. **Repo Awareness** - Complete project understanding
   - File Indexer: Semantic file scanning and indexing
   - Dependency Graph Builder: Code structure analysis
2. **Persistent Project State** - Session continuity
   - Project State Manager: State persistence across sessions
3. **Explainability** - Full transparency
   - Decision Logger: Complete reasoning audit trail
   - Execution Tracer: Ground-truth action timeline

### Phase 2: Governance & Routing (Complete)
1. **Git Governor** - Safe autonomous git operations
   - Branch Validator: Protected branch checks
   - Secret Detector: API key/password scanning
   - Risk Assessor: Change impact analysis
   - Git Governor: Unified governance engine
2. **Model Router** - Intelligent LLM routing
   - Task Classifier: Intent-based task categorization
   - Model Selector: Capability-based model selection
   - Cost Optimizer: Usage tracking and optimization
   - Model Router: Unified routing engine

---

## 📁 Directory Structure

```
a0-zero-v2/
├── README.md                    # This file
├── RESTORE.md                   # Restoration instructions
├── docs/
│   ├── AGENT_ZERO_V2_ARCHITECTURE.md  # Full architecture spec
│   └── phase2_summary.md         # Phase 2 completion report
│
├── phase1_subsystems/
│   ├── repo_awareness/          # File indexing & dependency graph
│   │   ├── file_indexer.py
│   │   └── dependency_graph.py
│   ├── project_state/           # State persistence
│   │   └── state_manager.py
│   ├── explainability/          # Decision & execution logging
│   │   ├── decision_logger.py
│   │   └── execution_tracer.py
│   └── phase1_schemas/          # Data contracts & tests
│       ├── interfaces.py
│       ├── control_contract.py
│       └── schemas.py
│
└── phase2_subsystems/
    ├── git_governor/            # Safe git operations
    │   ├── git_governor.py
    │   ├── commit_validator.py
    │   ├── secret_detector.py
    │   ├── branch_validator.py
    │   ├── approval_workflow.py
    │   └── test_git_governor.py
    └── model_router/            # Intelligent LLM routing
        ├── model_router.py
        ├── task_classifier.py
        ├── model_selector.py
        ├── cost_optimizer.py
        └── test_model_router.py
```

---

## 🚀 Quick Start (New Agent Zero Container)

### Option 1: Copy from GitHub
```bash
# In new Agent Zero container
cd /a0
rm -rf lib/bak
mv lib lib.bak
mkdir -p lib

# Clone repository (replace with your repo)
git clone https://github.com/tajo9128/cloudvina.git /tmp/repo

# Copy improvements
cp -r /tmp/repo/a0-zero-v2/phase1_subsystems/* lib/
cp -r /tmp/repo/a0-zero-v2/phase2_subsystems/* lib/

# Cleanup
rm -rf /tmp/repo

# Test installations
python -c "from lib.repo_awareness import FileIndexer; print('Phase 1 OK')"
python -c "from lib.phase2_subsystems.model_router import ModelRouter; print('Phase 2 OK')"
```

### Option 2: Manual Upload
1. Download `a0-zero-v2/` from GitHub
2. Upload to new Agent Zero container
3. Run: `cp -r a0-zero-v2/phase1_subsystems/* /a0/lib/`
4. Run: `cp -r a0-zero-v2/phase2_subsystems/* /a0/lib/`

---

## ✅ Test Installation

```bash
# Test Phase 1
cd /a0/lib/phase1_subsystems/phase1_schemas && python test_phase1.py

# Test Phase 2A (Git Governor)
cd /a0/lib/phase2_subsystems/git_governor && python test_git_governor.py

# Test Phase 2B (Model Router)
cd /a0/lib/phase2_subsystems/model_router && python test_model_router.py
```

**Expected Result**: All tests pass (100% success rate)

---

## 📊 Statistics

| Phase | Components | Tests | Status |
|-------|-----------|-------|--------|
| Phase 1 | 5 modules | 18/18 | ✅ Complete |
| Phase 2A | 7 modules | 16/16 | ✅ Complete |
| Phase 2B | 5 modules | 30/30 | ✅ Complete |
| **Total** | **17 modules** | **64/64** | **✅ 100% Pass** |

### Code Metrics
- **Production Code**: ~2,307 lines
- **Test Code**: ~829 lines
- **Documentation**: ~62,000 lines

---

## 🎯 Key Capabilities

### Repo Awareness
- ✅ Semantic file indexing with type detection
- ✅ Dependency graph with call chains and layers
- ✅ Incremental updates (only changed files)
- ✅ Query support by file type, function, class

### Git Governor
- ✅ Branch validation (protected branches, naming conventions)
- ✅ Secret detection (API keys, passwords, tokens)
- ✅ Risk assessment (file changes, merge conflicts)
- ✅ Approval workflows for high-risk changes
- ✅ Commit hygiene enforcement

### Model Router
- ✅ Task classification (PLANNING, CODE_GEN, DEBUG, TEST, DOCS, QUERY)
- ✅ Capability-based model selection (HIGH_QUALITY, CODE_SPECIALIZED, FAST_CHEAP)
- ✅ Cost optimization and usage tracking
- ✅ Model pool management

### Explainability
- ✅ Decision logging with reasoning and alternatives
- ✅ Execution tracing with full timeline
- ✅ Session vs project explainability separation

---

## 📝 Usage Examples

### Repo Awareness
```python
from lib.repo_awareness import FileIndexer, DependencyGraph

indexer = FileIndexer()
indexer.index("/a0/usr/projects")
results = indexer.query("type=python, pattern=async def")
```

### Git Governor
```python
from lib.phase2_subsystems.git_governor import GitGovernor

governor = GitGovernor()
result = governor.validate_commit(
    message="feat: Add new endpoint",
    files_modified=["api/endpoints.py"],
    branch_name="feature/new-api"
)
```

### Model Router
```python
from lib.phase2_subsystems.model_router import ModelRouter

router = ModelRouter()
decision = router.route("Generate a Python function")
print(f"Selected model: {decision.selection.model.name}")
```

---

## 🔧 Configuration

### Git Governor
Edit `/a0/lib/phase2_subsystems/git_governor/config.json`:
```json
{
  "protected_branches": ["main", "master", "dev"],
  "branch_patterns": ["^(feat|fix|docs|style|refactor|test|chore)/.+"],
  "secret_patterns": ["sk-.*", "api_key", "password"]
}
```

### Model Router
Edit `/a0/lib/phase2_subsystems/model_router/config.json`:
```json
{
  "default_models": {
    "HIGH_QUALITY": "gpt-4",
    "CODE_SPECIALIZED": "glm-4.7",
    "FAST_CHEAP": "mistral:latest"
  },
  "cost_threshold": 10.0
}
```

---

## 🐛 Troubleshooting

### Import Errors
```bash
# Ensure modules are in lib
ls -la /a0/lib/phase1_subsystems/
ls -la /a0/lib/phase2_subsystems/

# Check Python path
python -c "import sys; print(sys.path)"
```

### Test Failures
```bash
# Run individual test suites
cd /a0/lib/phase1_subsystems/phase1_schemas
python test_phase1.py

cd /a0/lib/phase2_subsystems/git_governor
python test_git_governor.py

cd /a0/lib/phase2_subsystems/model_router
python test_model_router.py
```

---

## 📚 Documentation

- **Architecture**: `docs/AGENT_ZERO_V2_ARCHITECTURE.md`
- **Phase 2 Summary**: `docs/phase2_summary.md`
- **Phase 1 Tests**: `phase1_subsystems/phase1_schemas/test_phase1.py`
- **Phase 2 Tests**: `phase2_subsystems/*/test_*.py`

---

## 🚀 Next Steps

### Planned (Phase 2C+)
- Git Governor + Model Router integration
- Open & Customizable plugin system
- Enterprise Policy engine (RBAC, compliance)

### Future Enhancements
- Real-time cost alerts
- Multi-repository git governance
- Dynamic model loading
- Auto-installation of git hooks

---

**Maintained in**: https://github.com/tajo9128/cloudvina/tree/main/a0-zero-v2

**Last Updated**: 2026-01-12

**Version**: v2.0.0 (Phase 2 Complete)

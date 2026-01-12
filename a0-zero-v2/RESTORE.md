# Restoration Guide - Agent Zero v2 IDE Improvements

## 🔄 Quick Restoration

### Prerequisites
- New Agent Zero container created
- GitHub repository cloned or accessible
- Docker container with root access

### Step 1: Backup Existing lib (Optional)
```bash
cd /a0
if [ -d "lib" ]; then
    mv lib lib.backup.$(date +%Y%m%d_%H%M%S)
fi
mkdir -p lib
```

### Step 2: Copy Phase 1
```bash
cp -r a0-zero-v2/phase1_subsystems/repo_awareness lib/
cp -r a0-zero-v2/phase1_subsystems/project_state lib/
cp -r a0-zero-v2/phase1_subsystems/explainability lib/
cp -r a0-zero-v2/phase1_subsystems/phase1_schemas lib/
```

### Step 3: Copy Phase 2
```bash
cp -r a0-zero-v2/phase2_subsystems/lib/
```

### Step 4: Verify Installation
```bash
# Test Phase 1
cd /a0/lib/phase1_subsystems/phase1_schemas
python test_phase1.py

# Test Phase 2A
cd /a0/lib/phase2_subsystems/git_governor
python test_git_governor.py

# Test Phase 2B
cd /a0/lib/phase2_subsystems/model_router
python test_model_router.py
```

### Step 5: Update Agent Zero Behavior (If Needed)

The improvements are automatically available once copied. Agent Zero will detect and use them.

---

## ✅ Verification Checklist

- [ ] Phase 1 subsystems copied to `/a0/lib/`
- [ ] Phase 2 subsystems copied to `/a0/lib/`
- [ ] All test suites pass (64/64 tests)
- [ ] Agent Zero recognizes new capabilities
- [ ] Repo Awareness functional
- [ ] Git Governor operational
- [ ] Model Router working

---

## 🆘 Troubleshooting

### ImportError: No module named 'lib'
```bash
# Ensure you're in the correct directory
cd /a0
python -c "import sys; sys.path.insert(0, '.'); from lib.repo_awareness import FileIndexer"
```

### Test Fails with "No such file or directory"
```bash
# Check file structure
ls -la /a0/lib/
ls -la /a0/lib/phase1_subsystems/
ls -la /a0/lib/phase2_subsystems/
```

### Phase 2 Import Errors
```bash
# Create init files if missing
touch /a0/lib/phase2_subsystems/__init__.py
touch /a0/lib/phase2_subsystems/git_governor/__init__.py
touch /a0/lib/phase2_subsystems/model_router/__init__.py
```

---

## 📞 Support

If issues persist:
1. Check test output for specific errors
2. Verify file permissions: `chmod -R 755 /a0/lib/`
3. Check Python version: `python --version` (requires 3.9+)
4. Review documentation: `README.md`

---
**Success**: Once all tests pass, your Agent Zero v2 is fully restored with IDE-grade capabilities!

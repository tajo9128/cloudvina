#!/bin/bash
# Restore Agent Zero v2 IDE improvements to /a0/lib/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
A0_LIB="/a0/lib"

# Backup existing lib
if [ -d "$A0_LIB" ]; then
    BACKUP_DIR="/a0/lib.backup.$(date +%Y%m%d_%H%M%S)"
    echo "Backing up existing lib to $BACKUP_DIR"
    cp -r $A0_LIB $BACKUP_DIR
fi

# Ensure target directory exists
mkdir -p $A0_LIB

# Copy Phase 1 subsystems
echo "Copying Phase 1 subsystems..."
cp -r "$SCRIPT_DIR/phase1_subsystems/repo_awareness" $A0_LIB/
cp -r "$SCRIPT_DIR/phase1_subsystems/project_state" $A0_LIB/
cp -r "$SCRIPT_DIR/phase1_subsystems/explainability" $A0_LIB/
cp -r "$SCRIPT_DIR/phase1_subsystems/phase1_schemas" $A0_LIB/

# Copy Phase 2 subsystems
echo "Copying Phase 2 subsystems..."
cp -r "$SCRIPT_DIR/phase2_subsystems/git_governor" $A0_LIB/phase2_subsystems/
cp -r "$SCRIPT_DIR/phase2_subsystems/model_router" $A0_LIB/phase2_subsystems/
cp "$SCRIPT_DIR/phase2_subsystems/__init__.py" $A0_LIB/phase2_subsystems/
cp "$SCRIPT_DIR/phase2_subsystems/PHASE2_PLAN.md" $A0_LIB/phase2_subsystems/

# Ensure phase2_subsystems exists in lib
mkdir -p $A0_LIB/phase2_subsystems

# Set permissions
chmod -R 755 $A0_LIB/phase1_subsystems
chmod -R 755 $A0_LIB/phase2_subsystems

echo ""
echo "✅ Agent Zero v2 IDE improvements restored successfully!"
echo ""
echo "Next steps:"
echo "1. cd $A0_LIB/phase1_subsystems/phase1_schemas && python test_phase1.py"
echo "2. cd $A0_LIB/phase2_subsystems/git_governor && python test_git_governor.py"
echo "3. cd $A0_LIB/phase2_subsystems/model_router && python test_model_router.py"

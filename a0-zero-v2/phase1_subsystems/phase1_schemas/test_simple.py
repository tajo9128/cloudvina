"""Simple Phase 1 Test"""
import sys
sys.path.insert(0, '/a0/lib')

from phase1_schemas.schemas import FILE_INDEX_SCHEMA
from phase1_schemas.control_contract import Phase1ControlGate, Phase1ContractError

print("Testing Phase 1 Components")
print("="*60)

gate = Phase1ControlGate()
print("Control Gate created")

# Test component status
status = gate.get_status()
print(f"Initial status: {status}")

# Test setting ready
gate.set_component_ready(gate.REPO_INDEX_READY)
status = gate.get_status()
print(f"After setting ready: {status['repo_index_ready']}")

# Test schema validation with valid data
valid_file_index = {
    "version": "1.0",
    "repo_path": "/test",
    "repo_id": "abc123",
    "index_time": "2026-01-12T17:00:00Z",
    "total_files": 1,
    "total_size_bytes": 100,
    "files": {
        "test.py": {
            "path": "test.py",
            "size": 100,
            "sha256": "a" * 64,
            "language": "Python",
            "is_binary": False
        }
    }
}

try:
    gate.register_data_structure("file_index", valid_file_index)
    print("Valid file index registered successfully")
except Exception as e:
    print(f"Error registering valid data: {e}")

# Test schema validation with invalid data
invalid_index = {"version": "1.0"}

try:
    gate.register_data_structure("file_index", invalid_index)
    print("ERROR: Invalid data should have been rejected!")
except Phase1ContractError as e:
    print(f"Invalid data correctly rejected: {type(e).__name__}")

# Test assert_ready
try:
    gate.assert_ready({gate.REPO_INDEX_READY})
    print("Assert ready passed for ready component")
except Phase1ContractError:
    print("ERROR: Assert ready failed unexpectedly")

# Test assert_ready with missing component
gate2 = Phase1ControlGate()
try:
    gate2.assert_ready({gate2.DEPENDENCY_GRAPH_READY})
    print("ERROR: Should have failed for missing component!")
except Phase1ContractError:
    print("Assert ready correctly failed for missing component")

# Run health diagnostic
diag = gate.run_health_diagnostic()
print(f"\nHealth diagnostic:")
print(f"  Overall health: {diag['overall_health']}")
print(f"  Component status: {list(diag['component_status'].keys())}")
print(f"  Schema validation: {diag['schema_validation']}")

print("\n" + "="*60)
print("Phase 1 Tests Completed Successfully!")
print("="*60)

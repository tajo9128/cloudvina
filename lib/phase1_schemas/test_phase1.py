"""
Comprehensive Phase 1 Test Suite

Tests all 4 subsystems:
1. Repo Awareness (File Indexer & Dependency Graph Builder)
2. Persistent State (Project State Manager)
3. Explainability (Decision Logger & Execution Tracer)
4. Control Contract (Phase1ControlGate)
"""

import sys
import os
import json
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from phase1_schemas.control_contract import Phase1ControlGate, Phase1ContractError
from phase1_schemas.schemas import (
    FILE_INDEX_SCHEMA,
    DEPENDENCY_GRAPH_SCHEMA,
    PROJECT_STATE_SCHEMA,
    DECISION_SCHEMA,
    EXECUTION_TRACE_SCHEMA,
    PROJECT_SNAPSHOT_SCHEMA
)


class Phase1TestHarness:
    """Test harness for Phase 1 components."""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.gate = Phase1ControlGate(str(project_path))
        self.test_results = []
    
    def log_result(self, test_name: str, passed: bool, details: str = ""):
        """Log a test result."""
        self.test_results.append({
            "test": test_name,
            "passed": passed,
            "details": details
        })
        status = "PASS" if passed else "FAIL"
        print(f"{status}: {test_name}")
        if details:
            print(f"     {details}")
    
    def test_repo_awareness(self):
        """Test Repo Awareness subsystem."""
        print("\n=== Testing Repo Awareness ===")
        
        # Create file index matching schema
        repo_id = hashlib.sha256(str(self.project_path).encode()).hexdigest()
        file_index = {
            "version": "1.0",
            "repo_path": str(self.project_path),
            "repo_id": repo_id,
            "index_time": "2025-01-12T17:00:00Z",
            "total_files": 1,
            "total_size_bytes": 1024,
            "files": {
                "test.py": {
                    "path": "test.py",
                    "type": "source",
                    "language": "python",
                    "size": 1024,
                    "hash": "abc123",
                    "last_modified": "2025-01-12T17:00:00Z",
                    "dependencies": ["os", "sys"]
                }
            }
        }
        
        try:
            self.gate.register_data_structure("file_index", file_index)
            self.gate.set_component_ready(self.gate.REPO_INDEX_READY)
            self.log_result("File Index Registration", True)
        except Exception as e:
            self.log_result("File Index Registration", False, str(e))
        
        # Create dependency graph matching schema
        dependency_graph = {
            "version": "1.0",
            "graph_time": "2025-01-12T17:00:00Z",
            "import_graph": {
                "nodes": ["test.py"],
                "edges": []
            },
            "runtime_graph": {
                "nodes": [],
                "edges": []
            },
            "test_graph": {
                "nodes": [],
                "edges": []
            }
        }
        
        try:
            self.gate.register_data_structure("dependency_graph", dependency_graph)
            self.gate.set_component_ready(self.gate.DEPENDENCY_GRAPH_READY)
            self.log_result("Dependency Graph Registration", True)
        except Exception as e:
            self.log_result("Dependency Graph Registration", False, str(e))
    
    def test_persistent_state(self):
        """Test Persistent State subsystem."""
        print("\n=== Testing Persistent State ===")
        
        # Create project state matching schema
        project_state = {
            "version": "1.0",
            "session_id": "test-session-001",
            "state_time": "2025-01-12T17:00:00Z",
            "state_hash": hashlib.sha256(b"test").hexdigest(),
            "is_tainted": False,
            "goal": {
                "description": "Run Phase 1 tests",
                "context": {"test_mode": True}
            },
            "steps": [
                {"step": 1, "description": "Initialize test harness"}
            ],
            "file_changes": [],
            "decisions": [],
            "risks": [
                {
                    "id": "R001",
                    "risk": "No tests found",
                    "status": "open",
                    "severity": "low"
                }
            ],
            "checkpoints": []
        }
        
        try:
            self.gate.register_data_structure("project_state", project_state)
            self.gate.set_component_ready(self.gate.STATE_LOADED)
            self.log_result("Project State Registration", True)
        except Exception as e:
            self.log_result("Project State Registration", False, str(e))
    
    def test_explainability(self):
        """Test Explainability subsystem."""
        print("\n=== Testing Explainability ===")
        
        # Create mock decision
        decision = {
            "id": "D001",
            "timestamp": "2025-01-12T17:00:00Z",
            "task": "Write test file",
            "decision": "Use Python unittest framework",
            "reasoning": [
                "Built-in to Python",
                "Simple syntax",
                "Good for unit tests"
            ],
            "alternatives": [
                {
                    "option": "pytest",
                    "reason_rejected": "External dependency"
                }
            ],
            "confidence": 0.85,
            "risk": "low",
            "outcome": "pending"
        }
        
        try:
            decisions_data = {"decisions": [decision]}
            self.gate.register_data_structure("decisions", decisions_data)
            self.gate.set_component_ready(self.gate.DECISION_LOGGER_READY)
            self.log_result("Decision Logger Registration", True)
        except Exception as e:
            self.log_result("Decision Logger Registration", False, str(e))
        
        # Create execution trace matching schema
        execution_trace = {
            "version": "1.0",
            "session_id": "test-session-001",
            "session_start": "2025-01-12T17:00:00Z",
            "execution_boundaries": {
                "max_files_modified": 3,
                "max_lines_changed": 200,
                "allow_new_files": True
            },
            "actions": [
                {
                    "step": 1,
                    "action": "Initialize test harness",
                    "tool": "code_execution_tool",
                    "input": {"file": "test_phase1.py"},
                    "output": "Success",
                    "status": "success",
                    "timestamp": "2025-01-12T17:00:00Z"
                }
            ]
        }
        
        try:
            self.gate.register_data_structure("execution_trace", execution_trace)
            self.gate.set_component_ready(self.gate.EXECUTION_TRACER_READY)
            self.log_result("Execution Tracer Registration", True)
        except Exception as e:
            self.log_result("Execution Tracer Registration", False, str(e))
    
    def test_control_contract(self):
        """Test Control Contract enforcement."""
        print("\n=== Testing Control Contract ===")
        
        # Test that all components are ready
        try:
            required = {
                self.gate.REPO_INDEX_READY,
                self.gate.DEPENDENCY_GRAPH_READY,
                self.gate.STATE_LOADED,
                self.gate.DECISION_LOGGER_READY,
                self.gate.EXECUTION_TRACER_READY
            }
            self.gate.assert_ready(required)
            self.log_result("All components ready", True)
        except Exception as e:
            self.log_result("All components ready", False, str(e))
        
        # Test schema validation
        schema_results = self.gate.validate_all_schemas()
        all_valid = all(schema_results.values())
        self.log_result("Schema validation", all_valid, str(schema_results))
    
    def test_health_diagnostic(self):
        """Test health diagnostic."""
        print("\n=== Testing Health Diagnostic ===")
        
        try:
            diagnostic = self.gate.run_health_diagnostic()
            print(json.dumps(diagnostic, indent=2))
            
            self.log_result(
                "Health diagnostic run",
                "overall_health" in diagnostic,
                "Health: " + diagnostic.get("overall_health", "unknown")
            )
        except Exception as e:
            self.log_result("Health diagnostic run", False, str(e))
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*50)
        print("PHASE 1 TEST SUMMARY")
        print("="*50)
        
        passed = sum(1 for r in self.test_results if r["passed"])
        total = len(self.test_results)
        
        print(f"Total tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        
        if failed_tests := [r for r in self.test_results if not r["passed"]]:
            print("\nFailed tests:")
            for test in failed_tests:
                print(f"  - {test['test']}: {test['details']}")
        
        success_rate = (passed / total * 100) if total > 0 else 0
        print(f"\nSuccess rate: {success_rate:.1f}%")
        print("="*50)


def main():
    """Run Phase 1 tests."""
    print("PHASE 1 COMPREHENSIVE TEST SUITE")
    print("="*50)
    
    project_path = "/a0/usr/projects/a0-cloudvina"
    harness = Phase1TestHarness(project_path)
    
    try:
        harness.test_repo_awareness()
        harness.test_persistent_state()
        harness.test_explainability()
        harness.test_control_contract()
        harness.test_health_diagnostic()
    except Exception as e:
        print(f"ERROR during tests: {e}")
        import traceback
        traceback.print_exc()
    
    harness.print_summary()
    return 0 if all(r["passed"] for r in harness.test_results) else 1


if __name__ == "__main__":
    sys.exit(main())

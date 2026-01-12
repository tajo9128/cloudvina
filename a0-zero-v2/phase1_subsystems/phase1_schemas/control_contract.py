"""
Phase 1 Control Contract - Implementation

The Master Gate that enforces all Phase 1 guarantees.
All Phase 1 components must pass through this gate before any agent action.
"""

import json
import hashlib
from typing import Dict, Set, Any, Optional
from pathlib import Path
from datetime import datetime

from .interfaces import IPhase1Control, Phase1ContractError
from .schemas import (
    FILE_INDEX_SCHEMA,
    DEPENDENCY_GRAPH_SCHEMA,
    PROJECT_STATE_SCHEMA,
    DECISION_SCHEMA,
    EXECUTION_TRACE_SCHEMA,
    PROJECT_SNAPSHOT_SCHEMA
)


class Phase1ControlGate(IPhase1Control):
    """
    Concrete implementation of Phase 1 Control Contract.
    
    This is the ENFORCER that ensures:
    - No execution before repo is indexed
    - No code modification before dependency graph is ready
    - No stateful operations before state is loaded
    - All data structures validate against schemas
    """
    
    def __init__(self, project_path: Optional[str] = None):
        self.project_path = project_path
        self._component_status: Dict[str, str] = {
            self.REPO_INDEX_READY: "not_ready",
            self.DEPENDENCY_GRAPH_READY: "not_ready",
            self.STATE_LOADED: "not_ready",
            self.DECISION_LOGGER_READY: "not_ready",
            self.EXECUTION_TRACER_READY: "not_ready"
        }
        self._data_structures: Dict[str, Optional[Dict]] = {
            "file_index": None,
            "dependency_graph": None,
            "project_state": None,
            "decisions": [],
            "execution_trace": None
        }
        self._schema_versions = {
            "file_index": "1.0",
            "dependency_graph": "1.0",
            "project_state": "1.0",
            "decision": "1.0",
            "execution_trace": "1.0",
            "snapshot": "1.0"
        }
    
    def set_component_ready(self, component: str) -> None:
        """Mark a component as ready."""
        if component not in self._component_status:
            raise ValueError(f"Unknown component: {component}")
        self._component_status[component] = "ready"
        print(f"Phase 1: {component} marked as ready")
    
    def set_component_partial(self, component: str, reason: str) -> None:
        """Mark a component as partial with reason."""
        if component not in self._component_status:
            raise ValueError(f"Unknown component: {component}")
        self._component_status[component] = f"partial: {reason}"
        print(f"Phase 1: {component} marked as partial: {reason}")
    
    def register_data_structure(self, name: str, data: Dict) -> None:
        """
        Register a data structure and validate it against its schema.
        """
        schema_map = {
            "file_index": FILE_INDEX_SCHEMA,
            "dependency_graph": DEPENDENCY_GRAPH_SCHEMA,
            "project_state": PROJECT_STATE_SCHEMA,
            "decisions": None,  # List of decision objects, validated individually
            "execution_trace": EXECUTION_TRACE_SCHEMA,
            "snapshot": PROJECT_SNAPSHOT_SCHEMA
        }
        
        if name not in schema_map:
            raise ValueError(f"Unknown data structure: {name}")
        
        schema = schema_map[name]
        
        if schema:
            is_valid, errors = self._validate_json_schema(data, schema)
            if not is_valid:
                error_list = [f"  - {e}" for e in errors]
                msg = f"Data structure '{name}' failed schema validation:\n"
                msg += "\n".join(error_list)
                raise Phase1ContractError(msg)
        
        self._data_structures[name] = data
        print(f"Phase 1: Data structure '{name}' registered and validated")
    
    def assert_ready(self, required_components: Set[str]) -> None:
        """
        Assert that required Phase 1 components are ready.
        
        Raises:
            Phase1ContractError if any component is not ready
        """
        not_ready = []
        partial = []
        
        for component in required_components:
            status = self._component_status.get(component, "not_found")
            if status == "not_ready" or status == "not_found":
                not_ready.append(component)
            elif status.startswith("partial"):
                partial.append(component)
        
        if not_ready:
            status_str = self._format_status()
            raise Phase1ContractError(
                f"Phase 1 Contract Violation: Required components not ready: {', '.join(not_ready)}\n{status_str}"
            )
        
        if partial:
            print(f"Phase 1 Warning: Partial components: {', '.join(partial)}")
        
        print(f"Phase 1 Gate: All required components ready: {', '.join(required_components)}")
    
    def validate_all_schemas(self) -> Dict[str, bool]:
        """
        Validate all Phase 1 data structures against schemas.
        
        Returns:
            Dict mapping data structure name to validation result
        """
        results = {}
        
        for name, data in self._data_structures.items():
            if data is None:
                results[name] = False
                continue
            
            schema_map = {
                "file_index": FILE_INDEX_SCHEMA,
                "dependency_graph": DEPENDENCY_GRAPH_SCHEMA,
                "project_state": PROJECT_STATE_SCHEMA,
                "execution_trace": EXECUTION_TRACE_SCHEMA,
                "snapshot": PROJECT_SNAPSHOT_SCHEMA
            }
            
            schema = schema_map.get(name)
            if schema is None:
                results[name] = True
                continue
            
            is_valid, errors = self._validate_json_schema(data, schema)
            results[name] = is_valid
            
            if not is_valid:
                print(f"Schema validation failed for {name}:")
                for error in errors:
                    print(f"  - {error}")
        
        return results
    
    def _validate_json_schema(self, data: Dict, schema: Dict) -> tuple[bool, list[str]]:
        """
        Validate data against JSON schema.
        
        Returns:
            (is_valid, list of errors)
        """
        errors = []
        
        required = schema.get("required", [])
        for field in required:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        properties = schema.get("properties", {})
        for field, field_schema in properties.items():
            if field not in data:
                continue
            
            value = data[field]
            field_type = field_schema.get("type")
            
            if field_type == "string":
                if not isinstance(value, str):
                    errors.append(f"Field '{field}' should be string, got {type(value).__name__}")
            elif field_type == "integer":
                if not isinstance(value, int):
                    errors.append(f"Field '{field}' should be integer, got {type(value).__name__}")
            elif field_type == "number":
                if not isinstance(value, (int, float)):
                    errors.append(f"Field '{field}' should be number, got {type(value).__name__}")
            elif field_type == "boolean":
                if not isinstance(value, bool):
                    errors.append(f"Field '{field}' should be boolean, got {type(value).__name__}")
            elif field_type == "array":
                if not isinstance(value, list):
                    errors.append(f"Field '{field}' should be array, got {type(value).__name__}")
            elif field_type == "object":
                if not isinstance(value, dict):
                    errors.append(f"Field '{field}' should be object, got {type(value).__name__}")
            
            if "const" in field_schema:
                if value != field_schema["const"]:
                    errors.append(f"Field '{field}' should be {field_schema['const']}, got {value}")
            
            if "minimum" in field_schema:
                if isinstance(value, (int, float)) and value < field_schema["minimum"]:
                    errors.append(f"Field '{field}' should be >= {field_schema['minimum']}, got {value}")
            
            if "maximum" in field_schema:
                if isinstance(value, (int, float)) and value > field_schema["maximum"]:
                    errors.append(f"Field '{field}' should be <= {field_schema['maximum']}, got {value}")
            
            if "enum" in field_schema:
                if value not in field_schema["enum"]:
                    errors.append(f"Field '{field}' should be one of {field_schema['enum']}, got {value}")
            
            if "pattern" in field_schema and isinstance(value, str):
                import re
                pattern = field_schema["pattern"]
                if pattern.startswith("r"):
                    pattern = pattern[1:]
                if not re.match(pattern, value):
                    errors.append(f"Field '{field}' does not match pattern {pattern}")
        
        return (len(errors) == 0, errors)
    
    def get_status(self) -> Dict[str, str]:
        """Get status of all Phase 1 components."""
        return self._component_status.copy()
    
    def run_health_diagnostic(self) -> Dict[str, Any]:
        """
        Run comprehensive Phase 1 health diagnostic.
        
        Returns:
            Diagnostic report with all component statuses
        """
        schema_validation = self.validate_all_schemas()
        
        not_ready = sum(1 for s in self._component_status.values() if s == "not_ready")
        partial = sum(1 for s in self._component_status.values() if s.startswith("partial"))
        invalid_schemas = sum(1 for v in schema_validation.values() if not v)
        
        open_risks = []
        state = self._data_structures.get("project_state")
        if state:
            for risk in state.get("risks", []):
                if risk.get("status") == "open":
                    open_risks.append(risk.get("risk", "Unknown"))
        
        last_decision = None
        decisions_data = self._data_structures.get("decisions", {})
        decisions = decisions_data.get("decisions", [])
        if decisions:
            last_decision = decisions[-1].get("id")
        
        overall_health = "healthy"
        if not_ready > 0 or invalid_schemas > 0:
            overall_health = "unhealthy"
        elif partial > 0 or open_risks:
            overall_health = "degraded"
        
        return {
            "overall_health": overall_health,
            "timestamp": datetime.now().isoformat(),
            "component_status": self._component_status,
            "schema_validation": schema_validation,
            "data_structures_loaded": {k: v is not None for k, v in self._data_structures.items()},
            "issues_summary": {
                "not_ready_components": not_ready,
                "partial_components": partial,
                "invalid_schemas": invalid_schemas,
                "open_risks": len(open_risks)
            },
            "last_decision": last_decision,
            "open_risks": open_risks,
            "schema_versions": self._schema_versions
        }
    
    def _format_status(self) -> str:
        """Format status for display."""
        lines = []
        for component, status in self._component_status.items():
            icon = "OK" if status == "ready" else "WARN" if status.startswith("partial") else "FAIL"
            lines.append(f"  {icon} {component}: {status}")
        return "\n".join(lines)

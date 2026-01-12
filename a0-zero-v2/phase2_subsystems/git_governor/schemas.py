"""
Git Governor - JSON Schemas for strict data validation.
All external communication must conform to these schemas.
"""
import json
from typing import Dict, Any
from hashlib import sha256

# Version for schema evolution
SCHEMA_VERSION = "2.0.0"


def calculate_checksum(data: Dict[str, Any]) -> str:
    """Calculate SHA256 checksum of dictionary data."""
    return sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


# Git Governor Schemas
GIT_GOVERNOR_SCHEMAS = {
    "commit_request": {
        "version": SCHEMA_VERSION,
        "type": "object",
        "required": ["files", "change_description", "commit_type"],
        "properties": {
            "version": {"type": "string", "const": SCHEMA_VERSION},
            "files": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "change_description": {"type": "string", "minLength": 10},
            "commit_type": {
                "type": "string",
                "enum": ["feat", "fix", "docs", "style", "refactor", "perf", "test", "chore"]
            },
            "scope": {"type": "string", "maxLength": 50},
            "body": {"type": "string"},
            "footer": {"type": "string"},
            "force": {"type": "boolean", "default": False}
        }
    },
    
    "commit_validation": {
        "version": SCHEMA_VERSION,
        "type": "object",
        "required": ["is_valid", "branch_valid", "secret_check", "risk_assessment", "can_proceed"],
        "properties": {
            "version": {"type": "string", "const": SCHEMA_VERSION},
            "is_valid": {"type": "boolean"},
            "branch_valid": {
                "type": "object",
                "required": ["is_valid", "branch_name", "is_protected", "can_commit"],
                "properties": {
                    "is_valid": {"type": "boolean"},
                    "branch_name": {"type": "string"},
                    "is_protected": {"type": "boolean"},
                    "error_message": {"type": "string"},
                    "can_commit": {"type": "boolean"}
                }
            },
            "secret_check": {
                "type": "object",
                "required": ["has_secrets", "secrets_found", "risk_level"],
                "properties": {
                    "has_secrets": {"type": "boolean"},
                    "secrets_found": {"type": "array", "items": {"type": "object"}},
                    "risk_level": {"type": "string", "enum": ["low", "medium", "high", "critical"]}
                }
            },
            "risk_assessment": {
                "type": "object",
                "required": ["overall_risk", "requires_approval", "confidence"],
                "properties": {
                    "overall_risk": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                    "risk_factors": {"type": "array", "items": {"type": "string"}},
                    "requires_approval": {"type": "boolean"},
                    "recommendations": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                }
            },
            "can_proceed": {"type": "boolean"},
            "errors": {"type": "array", "items": {"type": "string"}},
            "warnings": {"type": "array", "items": {"type": "string"}}
        }
    },
    
    "commit": {
        "version": SCHEMA_VERSION,
        "type": "object",
        "required": ["commit_hash", "branch", "message", "timestamp", "author"],
        "properties": {
            "version": {"type": "string", "const": SCHEMA_VERSION},
            "commit_hash": {"type": "string", "pattern": "^[a-f0-9]{40}$"},
            "branch": {"type": "string"},
            "message": {"type": "string", "minLength": 1},
            "timestamp": {"type": "string", "format": "date-time"},
            "author": {"type": "string"},
            "files": {"type": "array", "items": {"type": "string"}},
            "lines_changed": {"type": "object"},
            "checksum": {"type": "string"}
        }
    },
    
    "approval_request": {
        "version": SCHEMA_VERSION,
        "type": "object",
        "required": ["request_id", "commit_validation", "requested_by"],
        "properties": {
            "version": {"type": "string", "const": SCHEMA_VERSION},
            "request_id": {"type": "string", "format": "uuid"},
            "commit_validation": {"type": "object"},
            "requested_by": {"type": "string"},
            "requested_at": {"type": "string", "format": "date-time"},
            "expiry_time": {"type": "string", "format": "date-time"}
        }
    },
    
    "approval_response": {
        "version": SCHEMA_VERSION,
        "type": "object",
        "required": ["request_id", "status", "timestamp"],
        "properties": {
            "version": {"type": "string", "const": SCHEMA_VERSION},
            "request_id": {"type": "string"},
            "status": {"type": "string", "enum": ["pending", "approved", "rejected", "skipped"]},
            "approved_by": {"type": "string"},
            "approval_comment": {"type": "string"},
            "timestamp": {"type": "string", "format": "date-time"},
            "checksum": {"type": "string"}
        }
    },
    
    "checkpoint": {
        "version": SCHEMA_VERSION,
        "type": "object",
        "required": ["checkpoint_id", "label", "commit_hash", "timestamp"],
        "properties": {
            "version": {"type": "string", "const": SCHEMA_VERSION},
            "checkpoint_id": {"type": "string", "format": "uuid"},
            "label": {"type": "string", "minLength": 1},
            "commit_hash": {"type": "string"},
            "branch": {"type": "string"},
            "timestamp": {"type": "string", "format": "date-time"},
            "files_snapshot": {"type": "array", "items": {"type": "string"}},
            "checksum": {"type": "string"}
        }
    },
    
    "git_status": {
        "version": SCHEMA_VERSION,
        "type": "object",
        "required": ["branch", "is_clean"],
        "properties": {
            "version": {"type": "string", "const": SCHEMA_VERSION},
            "branch": {"type": "string"},
            "is_clean": {"type": "boolean"},
            "modified_files": {"type": "array", "items": {"type": "string"}},
            "staged_files": {"type": "array", "items": {"type": "string"}},
            "untracked_files": {"type": "array", "items": {"type": "string"}},
            "ahead_commits": {"type": "integer", "minimum": 0},
            "behind_commits": {"type": "integer", "minimum": 0},
            "last_commit": {"type": "string"}
        }
    },
    
    "push_result": {
        "version": SCHEMA_VERSION,
        "type": "object",
        "required": ["success", "remote", "branch"],
        "properties": {
            "version": {"type": "string", "const": SCHEMA_VERSION},
            "success": {"type": "boolean"},
            "remote": {"type": "string"},
            "branch": {"type": "string"},
            "commit_hash": {"type": "string"},
            "ci_status": {"type": "string"},
            "error_message": {"type": "string"},
            "checksum": {"type": "string"}
        }
    }
}


def validate_schema(schema_name: str, data: Dict[str, Any]) -> tuple[bool, list]:
    """Validate data against schema."""
    import jsonschema
    
    if schema_name not in GIT_GOVERNOR_SCHEMAS:
        return False, [f"Schema '{schema_name}' not found"]
    
    schema = GIT_GOVERNOR_SCHEMAS[schema_name]
    
    # Check version
    if "version" in schema and data.get("version") != schema["version"]:
        return False, [f"Version mismatch: expected {schema['version']}, got {data.get('version')}"]
    
    # Validate with jsonschema
    validator = jsonschema.Draft7Validator(schema)
    errors = []
    
    for error in validator.iter_errors(data):
        errors.append(f"{error.path}: {error.message}")
    
    # Validate checksum if present
    if "checksum" in data:
        data_copy = data.copy()
        checksum_stored = data_copy.pop("checksum")
        checksum_calculated = calculate_checksum(data_copy)
        if checksum_stored != checksum_calculated:
            errors.append(f"Checksum mismatch: stored={checksum_stored}, calculated={checksum_calculated}")
    
    return len(errors) == 0, errors


def add_checksum(data: Dict[str, Any]) -> Dict[str, Any]:
    """Add version and checksum to data."""
    result = data.copy()
    result["version"] = SCHEMA_VERSION
    result["checksum"] = calculate_checksum(result)
    return result

"""
Phase 1 - JSON Schemas

Strict JSON Schema definitions for all Phase 1 data structures.
All Phase 1 components MUST validate against these schemas.

Using JSON Schema Draft 2020-12 format.
"""

# =============================================================================
# FILE INDEX SCHEMA
# =============================================================================

FILE_INDEX_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "File Index",
    "description": "Complete index of all files in a repository",
    "type": "object",
    "required": [
        "version",
        "repo_path",
        "repo_id",
        "index_time",
        "total_files",
        "total_size_bytes",
        "files"
    ],
    "properties": {
        "version": {
            "type": "string",
            "const": "1.0"
        },
        "repo_path": {
            "type": "string",
            "description": "Absolute path to repository root"
        },
        "repo_id": {
            "type": "string",
            "description": "SHA256 hash of repository root path (for change detection)"
        },
        "index_time": {
            "type": "string",
            "format": "date-time",
            "description": "ISO 8601 timestamp of when index was created"
        },
        "total_files": {
            "type": "integer",
            "minimum": 0
        },
        "total_size_bytes": {
            "type": "integer",
            "minimum": 0
        },
        "file_stats": {
            "type": "object",
            "description": "Statistics by file extension",
            "additionalProperties": {
                "type": "integer"
            }
        },
        "files": {
            "type": "object",
            "description": "Map of relative file paths to file metadata",
            "additionalProperties": {
                "type": "object",
                "required": ["path", "size", "sha256", "language"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path from repo root"
                    },
                    "absolute_path": {
                        "type": "string",
                        "description": "Absolute file path"
                    },
                    "size": {
                        "type": "integer",
                        "minimum": 0
                    },
                    "sha256": {
                        "type": "string",
                        "pattern": r"^[a-f0-9]{64}$",
                        "description": "SHA256 hash of file contents"
                    },
                    "language": {
                        "type": "string",
                        "description": "Detected programming language"
                    },
                    "last_modified": {
                        "type": "string",
                        "format": "date-time"
                    },
                    "is_binary": {
                        "type": "boolean"
                    }
                }
            }
        }
    }
}

# =============================================================================
# DEPENDENCY GRAPH SCHEMA (with logical layers)
# =============================================================================

DEPENDENCY_GRAPH_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Dependency Graph",
    "description": "Directed graph of file dependencies separated into logical layers",
    "type": "object",
    "required": [
        "version",
        "graph_time",
        "import_graph",
        "runtime_graph",
        "test_graph"
    ],
    "properties": {
        "version": {
            "type": "string",
            "const": "1.0"
        },
        "graph_time": {
            "type": "string",
            "format": "date-time"
        },
        "import_graph": {
            "$ref": "#/definitions/graph_layer"
        },
        "runtime_graph": {
            "$ref": "#/definitions/graph_layer"
        },
        "test_graph": {
            "$ref": "#/definitions/graph_layer"
        },
        "critical_files": {
            "type": "array",
            "description": "Files sorted by dependency count (bottlenecks)",
            "items": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "upstream_count": {"type": "integer"},
                    "downstream_count": {"type": "integer"}
                }
            }
        }
    },
    "definitions": {
        "graph_layer": {
            "type": "object",
            "description": "A single dependency graph layer",
            "properties": {
                "nodes": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "edges": {
                    "type": "object",
                    "description": "Map of source file to list of dependencies",
                    "additionalProperties": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "file": {"type": "string"},
                                "import_type": {"type": "string", "enum": ["direct", "indirect"]},
                                "module_name": {"type": "string"}
                            }
                        }
                    }
                }
            }
        }
    }
}

# =============================================================================
# PROJECT STATE SCHEMA (with integrity checksums)
# =============================================================================

PROJECT_STATE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Project State",
    "description": "Persistent state tracking across sessions with integrity checks",
    "type": "object",
    "required": [
        "version",
        "session_id",
        "state_time",
        "state_hash",
        "is_tainted",
        "goal",
        "steps",
        "file_changes",
        "decisions",
        "risks",
        "checkpoints"
    ],
    "properties": {
        "version": {
            "type": "string",
            "const": "1.0"
        },
        "session_id": {
            "type": "string"
        },
        "state_time": {
            "type": "string",
            "format": "date-time"
        },
        "state_hash": {
            "type": "string",
            "description": "SHA256 hash of state contents (excluding this field)"
        },
        "is_tainted": {
            "type": "boolean",
            "description": "True if integrity check failed"
        },
        "goal": {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "context": {"type": "object"}
            }
        },
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step": {"type": "string"},
                    "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "failed", "skipped"]},
                    "time": {"type": "string", "format": "date-time"}
                }
            }
        },
        "file_changes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "change_type": {"type": "string", "enum": ["added", "modified", "deleted", "moved"]},
                    "old_hash": {"type": "string"},
                    "new_hash": {"type": "string"},
                    "reason": {"type": "string"},
                    "time": {"type": "string", "format": "date-time"}
                }
            }
        },
        "decisions": {
            "type": "array",
            "items": {"$ref": "#/definitions/decision_ref"}
        },
        "risks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "risk": {"type": "string"},
                    "mitigation": {"type": "string"},
                    "level": {"type": "string", "enum": ["none", "low", "medium", "high", "critical"]},
                    "status": {"type": "string", "enum": ["open", "mitigated", "accepted", "closed"]}
                }
            }
        },
        "checkpoints": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "checkpoint_id": {"type": "string"},
                    "description": {"type": "string"},
                    "time": {"type": "string", "format": "date-time"},
                    "step_index": {"type": "integer"},
                    "state_snapshot": {"type": "object"}
                }
            }
        }
    },
    "definitions": {
        "decision_ref": {
            "type": "object",
            "properties": {
                "decision_id": {"type": "string"},
                "decision": {"type": "string"},
                "time": {"type": "string", "format": "date-time"}
            }
        }
    }
}

# =============================================================================
# DECISION SCHEMA (with confidence scoring)
# =============================================================================

DECISION_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Decision",
    "description": "A single agent decision with full reasoning context",
    "type": "object",
    "required": [
        "id",
        "type",
        "intent",
        "decision",
        "reasoning",
        "confidence",
        "risk_level",
        "timestamp"
    ],
    "properties": {
        "id": {
            "type": "string",
            "pattern": r"^dec_\d{8}_\d{6}_\d+$"
        },
        "type": {
            "type": "string",
            "enum": [
                "plan", "tool_selection", "model_selection", "refactoring",
                "algorithm", "architecture", "workflow", "risk_acceptance",
                "rollback", "pivot", "other"
            ]
        },
        "intent": {
            "type": "string",
            "minLength": 1
        },
        "decision": {
            "type": "string",
            "minLength": 1
        },
        "reasoning": {
            "type": "string",
            "minLength": 1
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Confidence score (0.0 to 1.0)"
        },
        "risk_level": {
            "type": "string",
            "enum": ["none", "low", "medium", "high", "critical"]
        },
        "alternatives": {
            "type": "array",
            "items": {"type": "string"}
        },
        "rejected_reasons": {
            "type": "array",
            "items": {"type": "string"}
        },
        "model_used": {
            "type": "string"
        },
        "context": {
            "type": "object"
        },
        "timestamp": {
            "type": "string",
            "format": "date-time"
        },
        "session_time_seconds": {
            "type": "number",
            "minimum": 0
        }
    }
}

# =============================================================================
# EXECUTION TRACE SCHEMA (with boundaries)
# =============================================================================

EXECUTION_TRACE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Execution Trace",
    "description": "Complete execution timeline with safety boundaries",
    "type": "object",
    "required": [
        "version",
        "session_id",
        "session_start",
        "execution_boundaries",
        "actions"
    ],
    "properties": {
        "version": {
            "type": "string",
            "const": "1.0"
        },
        "session_id": {
            "type": "string"
        },
        "session_start": {
            "type": "string",
            "format": "date-time"
        },
        "execution_boundaries": {
            "type": "object",
            "description": "Safety boundaries for this session",
            "properties": {
                "max_files_modified": {"type": "integer", "minimum": 0},
                "max_lines_changed": {"type": "integer", "minimum": 0},
                "allow_new_files": {"type": "boolean"},
                "boundary_exceeded": {"type": "boolean"}
            }
        },
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "type", "tool", "inputs", "status", "start_time"],
                "properties": {
                    "id": {
                        "type": "string",
                        "pattern": r"^act_\d{8}_\d{6}_\d+$"
                    },
                    "type": {
                        "type": "string",
                        "enum": [
                            "tool_call", "file_read", "file_write", "file_delete",
                            "code_execute", "api_call", "git_operation",
                            "subordinate_call", "memory_operation",
                            "user_interaction", "system_command", "other"
                        ]
                    },
                    "tool": {"type": "string"},
                    "inputs": {"type": "object"},
                    "outputs": {"type": "object"},
                    "status": {
                        "type": "string",
                        "enum": ["started", "success", "failed", "partial", "rolled_back", "skipped"]
                    },
                    "start_time": {"type": "string", "format": "date-time"},
                    "end_time": {"type": "string", "format": "date-time"},
                    "duration_seconds": {"type": "number", "minimum": 0},
                    "side_effects": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "path": {"type": "string"},
                                "content": {"type": "string"},
                                "new_hash": {"type": "string"},
                                "old_hash": {"type": "string"}
                            }
                        }
                    },
                    "errors": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "context": {"type": "object"},
                    "parent_id": {"type": "string"},
                    "children": {"type": "array", "items": {"type": "string"}}
                }
            }
        }
    }
}

# =============================================================================
# PROJECT SNAPSHOT SCHEMA (for versioning)
# =============================================================================

PROJECT_SNAPSHOT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Project Snapshot",
    "description": "Repository snapshot metadata for versioning and corruption detection",
    "type": "object",
    "required": [
        "version",
        "repo_id",
        "snapshot_id",
        "snapshot_time",
        "agent_zero_version"
    ],
    "properties": {
        "version": {
            "type": "string",
            "const": "1.0"
        },
        "repo_id": {
            "type": "string",
            "description": "SHA256 hash of repository root path"
        },
        "snapshot_id": {
            "type": "string",
            "description": "SHA256 hash of file index (for change detection)"
        },
        "snapshot_time": {
            "type": "string",
            "format": "date-time"
        },
        "agent_zero_version": {
            "type": "string",
            "description": "Agent Zero version that created this snapshot"
        },
        "previous_snapshot_id": {
            "type": "string",
            "description": "Previous snapshot ID (if this is an update)"
        },
        "schema_versions": {
            "type": "object",
            "description": "Schema versions used for this snapshot",
            "additionalProperties": {"type": "string"}
        }
    }
}

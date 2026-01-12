"""
Phase 1 - Strict Interfaces and JSON Schemas

This module defines the contract for all Phase 1 components:
- File Indexer
- Dependency Graph
- Project State
- Decision Logger
- Execution Tracer

All Phase 1 components must implement these interfaces.
All data structures must validate against these schemas.
"""

from .interfaces import (
    IFileIndexer,
    IDependencyGraph,
    IProjectStateManager,
    IDecisionLogger,
    IExecutionTracer,
    IPhase1Control
)

from .schemas import (
    FILE_INDEX_SCHEMA,
    DEPENDENCY_GRAPH_SCHEMA,
    PROJECT_STATE_SCHEMA,
    DECISION_SCHEMA,
    EXECUTION_TRACE_SCHEMA,
    PROJECT_SNAPSHOT_SCHEMA
)

__all__ = [
    'IFileIndexer',
    'IDependencyGraph',
    'IProjectStateManager',
    'IDecisionLogger',
    'IExecutionTracer',
    'IPhase1Control',
    'FILE_INDEX_SCHEMA',
    'DEPENDENCY_GRAPH_SCHEMA',
    'PROJECT_STATE_SCHEMA',
    'DECISION_SCHEMA',
    'EXECUTION_TRACE_SCHEMA',
    'PROJECT_SNAPSHOT_SCHEMA'
]

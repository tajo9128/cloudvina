"""
Phase 1 - Strict Interfaces

These interfaces define the contract that all Phase 1 components MUST implement.
Violating these interfaces will cause Phase 1 Control Contract failures.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
from datetime import datetime


class IFileIndexer(ABC):
    """
    Interface for File System Indexer.

    Contract:
    - Must index entire repository
    - Must compute SHA256 hashes for all files
    - Must detect incremental changes
    - Must validate against FILE_INDEX_SCHEMA
    """

    @abstractmethod
    def index_repository(self, repo_path: str) -> Dict[str, Any]:
        """
        Index entire repository.

        Returns:
            dict matching FILE_INDEX_SCHEMA
        """
        pass

    @abstractmethod
    def get_changed_files(self, since_index: Dict[str, Any]) -> Dict[str, Dict]:
        """
        Get files changed since last index.

        Args:
            since_index: Previous file index

        Returns:
            Dict of changed files with status (added, modified, deleted)
        """
        pass

    @abstractmethod
    def validate_index(self, index: Dict[str, Any]) -> bool:
        """
        Validate index matches FILE_INDEX_SCHEMA.

        Returns:
            True if valid, False otherwise
        """
        pass


class IDependencyGraph(ABC):
    """
    Interface for Dependency Graph Builder.

    Contract:
    - Must parse Python and JavaScript imports
    - Must build directed dependency graph
    - Must separate into logical layers (import, runtime, test)
    - Must validate against DEPENDENCY_GRAPH_SCHEMA
    """

    @abstractmethod
    def build_graph(self, file_index: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build dependency graph from file index.

        Args:
            file_index: File index from FileIndexer

        Returns:
            dict matching DEPENDENCY_GRAPH_SCHEMA
        """
        pass

    @abstractmethod
    def get_upstream_deps(self, file_path: str) -> List[str]:
        """
        Get all upstream dependencies for a file.

        Args:
            file_path: Path to file

        Returns:
            List of upstream file paths
        """
        pass

    @abstractmethod
    def get_downstream_impact(self, file_path: str) -> List[str]:
        """
        Get all downstream files affected by this file.

        Args:
            file_path: Path to file

        Returns:
            List of downstream file paths
        """
        pass

    @abstractmethod
    def get_critical_files(self) -> List[str]:
        """
        Get files with highest dependency count (bottlenecks).

        Returns:
            List of file paths sorted by dependency count
        """
        pass


class IProjectStateManager(ABC):
    """
    Interface for Project State Manager.

    Contract:
    - Must track goals, steps, file changes, decisions, risks
    - Must support checkpoints and rollback
    - Must validate state integrity (checksums)
    - Must validate against PROJECT_STATE_SCHEMA
    """

    @abstractmethod
    def set_goal(self, goal: str, context: Optional[Dict] = None) -> None:
        """
        Set current project goal.
        """
        pass

    @abstractmethod
    def add_step(self, step: str, status: str) -> None:
        """
        Add a workflow step.
        """
        pass

    @abstractmethod
    def record_file_change(self, file_path: str, change_type: str, 
                          new_hash: str, reason: str) -> None:
        """
        Record a file change.
        """
        pass

    @abstractmethod
    def record_decision(self, decision: str, reasoning: str, 
                       alternatives: List[str]) -> None:
        """
        Record a decision.
        """
        pass

    @abstractmethod
    def record_risk(self, risk: str, mitigation: str, level: str) -> None:
        """
        Record a risk.
        """
        pass

    @abstractmethod
    def create_checkpoint(self, description: str) -> str:
        """
        Create a checkpoint.

        Returns:
            Checkpoint ID
        """
        pass

    @abstractmethod
    def rollback_to_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Rollback to a checkpoint.

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def verify_state_integrity(self) -> bool:
        """
        Verify state integrity using checksums.

        Returns:
            True if state is valid, False if tainted
        """
        pass

    @abstractmethod
    def get_progress(self) -> Dict[str, Any]:
        """
        Get current progress.

        Returns:
            Progress dict with completion percentage
        """
        pass


class IDecisionLogger(ABC):
    """
    Interface for Decision Logger.

    Contract:
    - Must log all decisions with intent, reasoning, alternatives
    - Must include confidence scores (0.0 to 1.0)
    - Must assess risk levels
    - Must validate against DECISION_SCHEMA
    """

    @abstractmethod
    def log_decision(self, intent: str, decision: str, reasoning: str,
                     confidence: float, risk_level: str,
                     alternatives: Optional[List[str]] = None) -> str:
        """
        Log a decision with full reasoning.

        Args:
            intent: What was the agent trying to achieve
            decision: What was chosen
            reasoning: Why this decision was made
            confidence: Confidence level (0.0 to 1.0)
            risk_level: Risk level (none, low, medium, high, critical)
            alternatives: What other options were considered

        Returns:
            Decision ID
        """
        pass

    @abstractmethod
    def get_low_confidence_decisions(self, max_confidence: float = 0.3) -> List[Dict]:
        """
        Get decisions with low confidence.
        """
        pass

    @abstractmethod
    def get_high_risk_decisions(self, min_risk: str = "high") -> List[Dict]:
        """
        Get decisions at or above a risk level.
        """
        pass

    @abstractmethod
    def generate_audit_report(self) -> str:
        """
        Generate audit report of all decisions.
        """
        pass


class IExecutionTracer(ABC):
    """
    Interface for Execution Tracer.

    Contract:
    - Must trace all actions with inputs/outputs/status
    - Must enforce execution boundaries
    - Must track side effects
    - Must validate against EXECUTION_TRACE_SCHEMA
    """

    @abstractmethod
    def start_action(self, action_type: str, tool: str, inputs: Dict) -> str:
        """
        Start tracing an action.

        Returns:
            Action ID
        """
        pass

    @abstractmethod
    def end_action(self, action_id: str, status: str, outputs: Optional[Dict],
                  side_effects: Optional[List[Dict]], errors: Optional[List[str]]) -> None:
        """
        End tracing an action.
        """
        pass

    @abstractmethod
    def set_execution_boundaries(self, max_files: int, max_lines: int, 
                                 allow_new: bool) -> None:
        """
        Set execution boundaries for safety.

        Args:
            max_files: Maximum files that can be modified
            max_lines: Maximum lines that can be changed
            allow_new: Whether new files can be created
        """
        pass

    @abstractmethod
    def check_boundary_exceeded(self) -> bool:
        """
        Check if execution boundaries have been exceeded.

        Returns:
            True if exceeded, False otherwise
        """
        pass

    @abstractmethod
    def get_failed_actions(self) -> List[Dict]:
        """
        Get all failed actions.
        """
        pass

    @abstractmethod
    def generate_execution_report(self) -> str:
        """
        Generate execution report.
        """
        pass


class IPhase1Control(ABC):
    """
    Phase 1 Control Contract - The Master Gate.

    This is the ENFORCER of Phase 1 guarantees.
    All components must pass this gate before any agent action.

    Contract:
    - Must enforce that repo_index is ready before any action
    - Must enforce that dependency_graph is ready before any code modification
    - Must enforce that project_state is loaded before any stateful operation
    - Must validate all data structures against schemas
    - Must raise exceptions if contracts are violated
    """

    # Required Phase 1 components
    REPO_INDEX_READY = "repo_index_ready"
    DEPENDENCY_GRAPH_READY = "dependency_graph_ready"
    STATE_LOADED = "state_loaded"
    DECISION_LOGGER_READY = "decision_logger_ready"
    EXECUTION_TRACER_READY = "execution_tracer_ready"

    @abstractmethod
    def assert_ready(self, required_components: Set[str]) -> None:
        """
        Assert that required Phase 1 components are ready.

        Args:
            required_components: Set of required component names

        Raises:
            Phase1ContractError if any component is not ready
        """
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, str]:
        """
        Get status of all Phase 1 components.

        Returns:
            Dict mapping component name to status (ready, partial, not_ready)
        """
        pass

    @abstractmethod
    def validate_all_schemas(self) -> Dict[str, bool]:
        """
        Validate all Phase 1 data structures against schemas.

        Returns:
            Dict mapping data structure name to validation result
        """
        pass

    @abstractmethod
    def run_health_diagnostic(self) -> Dict[str, Any]:
        """
        Run comprehensive Phase 1 health diagnostic.

        Returns:
            Diagnostic report with all component statuses
        """
        pass


class Phase1ContractError(Exception):
    """
    Raised when Phase 1 Control Contract is violated.

    This should stop execution immediately.
    """
    pass

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum

"""
Phase 1.3: Persistent Project State Engine

Purpose: Make Agent Zero stateful across time, not just within a prompt
Agentic Impact: Enables pause/resume, long-running tasks, prevents repeated work, required for self-correction
"""


class TaskStatus(Enum):
    """Status of a task or step in the project."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class RiskLevel(Enum):
    """Risk level for project decisions."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ProjectStateManager:
    """
    Manages persistent state for project work sessions.

    Tracks:
    - Current goal and context
    - Completed steps and remaining work
    - Files modified and changes made
    - Decisions made with reasoning
    - Known risks and mitigations
    - Checkpoints for rollback
    - Session history
    """

    def __init__(self, project_path: str, state_dir: Optional[str] = None):
        """
        Initialize ProjectStateManager for a project.

        Args:
            project_path: Root directory of the project
            state_dir: Directory to store state files (default: .a0proj/)
        """
        self.project_path = Path(project_path).resolve()

        if state_dir:
            self.state_dir = Path(state_dir).resolve()
        else:
            self.state_dir = self.project_path / ".a0proj"

        # State data
        self.goal: Optional[str] = None
        self.context: Dict[str, Any] = {}
        self.completed_steps: List[str] = []
        self.current_step: Optional[str] = None
        self.files_modified: List[Dict] = []
        self.decisions: List[Dict] = []
        self.risks: List[Dict] = []
        self.checkpoints: List[Dict] = []
        self.session_history: List[Dict] = []
        self.created_at: datetime = datetime.now()
        self.updated_at: datetime = datetime.now()
        self.last_checkpoint: Optional[datetime] = None

    def set_goal(self, goal: str, context: Optional[Dict] = None) -> None:
        """
        Set the current goal for the project.

        Args:
            goal: Description of the goal
            context: Additional context information
        """
        self.goal = goal
        self.context = context or {}
        self.updated_at = datetime.now()

    def add_step(self, step: str, status: TaskStatus = TaskStatus.PENDING) -> None:
        """
        Add a step to the project workflow.

        Args:
            step: Description of the step
            status: Initial status of the step
        """
        self.session_history.append({
            "action": "add_step",
            "step": step,
            "status": status.value,
            "timestamp": datetime.now().isoformat()
        })
        self.updated_at = datetime.now()

        if status == TaskStatus.IN_PROGRESS:
            self.current_step = step
        elif status == TaskStatus.COMPLETED:
            self._complete_step(step)

    def _complete_step(self, step: str) -> None:
        """
        Mark a step as completed.

        Args:
            step: Step to mark complete
        """
        if step not in self.completed_steps:
            self.completed_steps.append(step)

        if self.current_step == step:
            self.current_step = None

        self.updated_at = datetime.now()

    def record_file_change(self, file_path: str, change_type: str, 
                           old_hash: Optional[str] = None, 
                           new_hash: Optional[str] = None,
                           reason: Optional[str] = None) -> None:
        """
        Record a file modification.

        Args:
            file_path: Relative path to the file
            change_type: Type of change (modified, added, deleted)
            old_hash: Previous file hash
            new_hash: New file hash
            reason: Reason for the change
        """
        change = {
            "file": file_path,
            "type": change_type,
            "timestamp": datetime.now().isoformat(),
            "old_hash": old_hash,
            "new_hash": new_hash,
            "reason": reason
        }

        self.files_modified.append(change)
        self.updated_at = datetime.now()

    def record_decision(self, decision: str, reasoning: str,
                       alternatives: Optional[List[str]] = None,
                       risk: RiskLevel = RiskLevel.MEDIUM) -> None:
        """
        Record a decision made during the project.

        Args:
            decision: The decision made
            reasoning: Why this decision was made
            alternatives: Other options considered
            risk: Risk level of this decision
        """
        decision_entry = {
            "decision": decision,
            "reasoning": reasoning,
            "alternatives": alternatives or [],
            "risk": risk.value,
            "timestamp": datetime.now().isoformat()
        }

        self.decisions.append(decision_entry)
        self.updated_at = datetime.now()

    def record_risk(self, risk: str, mitigation: Optional[str] = None,
                    level: RiskLevel = RiskLevel.MEDIUM) -> None:
        """
        Record a known risk.

        Args:
            risk: Description of the risk
            mitigation: How to mitigate the risk
            level: Risk level
        """
        risk_entry = {
            "risk": risk,
            "mitigation": mitigation,
            "level": level.value,
            "open": True,
            "timestamp": datetime.now().isoformat()
        }

        self.risks.append(risk_entry)
        self.updated_at = datetime.now()

    def close_risk(self, risk_index: int, resolution: str) -> None:
        """
        Mark a risk as resolved.

        Args:
            risk_index: Index of the risk in self.risks list
            resolution: How the risk was resolved
        """
        if 0 <= risk_index < len(self.risks):
            self.risks[risk_index]["open"] = False
            self.risks[risk_index]["resolution"] = resolution
            self.risks[risk_index]["resolved_at"] = datetime.now().isoformat()
            self.updated_at = datetime.now()

    def create_checkpoint(self, description: str) -> str:
        """
        Create a checkpoint for rollback.

        Args:
            description: Description of the checkpoint

        Returns:
            Checkpoint ID
        """
        checkpoint_id = f"cp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        checkpoint = {
            "id": checkpoint_id,
            "description": description,
            "timestamp": datetime.now().isoformat(),
            "step": self.current_step,
            "completed_steps": list(self.completed_steps),
            "files_at_checkpoint": len(self.files_modified),
            "decisions_at_checkpoint": len(self.decisions)
        }

        self.checkpoints.append(checkpoint)
        self.last_checkpoint = datetime.now()
        self.updated_at = datetime.now()

        return checkpoint_id

    def rollback_to_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Rollback to a previous checkpoint (metadata only).

        Note: Actual file rollback would need to be implemented separately
        with file backups.

        Args:
            checkpoint_id: ID of checkpoint to rollback to

        Returns:
            True if rollback successful
        """
        checkpoint = None
        for cp in self.checkpoints:
            if cp["id"] == checkpoint_id:
                checkpoint = cp
                break

        if not checkpoint:
            return False

        # Restore state from checkpoint
        self.completed_steps = list(checkpoint["completed_steps"])
        self.current_step = checkpoint["step"]

        # Remove files and decisions after checkpoint
        # (This is metadata only - actual file rollback needed)
        checkpoint_idx = self.checkpoints.index(checkpoint)
        # In a full implementation, we'd track which files/decisions
        # belong to each checkpoint

        self.updated_at = datetime.now()

        return True

    def get_progress(self) -> Dict[str, Any]:
        """
        Get current progress information.

        Returns:
            Progress dict with completion metrics
        """
        total_steps = len(self.completed_steps) + (1 if self.current_step else 0)
        completion_percentage = 0

        if total_steps > 0:
            completion_percentage = (len(self.completed_steps) / total_steps) * 100

        open_risks = [r for r in self.risks if r["open"]]

        return {
            "goal": self.goal,
            "current_step": self.current_step,
            "completed_steps": len(self.completed_steps),
            "completion_percentage": round(completion_percentage, 2),
            "files_modified": len(self.files_modified),
            "decisions_made": len(self.decisions),
            "open_risks": len(open_risks),
            "last_checkpoint": self.last_checkpoint.isoformat() if self.last_checkpoint else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    def get_remaining_work(self) -> List[str]:
        """
        Get a summary of remaining work.

        Returns:
            List of remaining work items
        """
        # Analyze session history to find pending steps
        remaining = []

        for entry in self.session_history:
            if entry.get("action") == "add_step":
                step = entry["step"]
                status = entry.get("status", TaskStatus.PENDING.value)

                if status in [TaskStatus.PENDING.value, TaskStatus.BLOCKED.value]:
                    if step not in self.completed_steps and step != self.current_step:
                        remaining.append(f"{step} (status: {status})")

        return remaining

    def save_state(self, state_file: Optional[str] = None) -> None:
        """
        Save project state to JSON file.

        Args:
            state_file: Path to save state (default: .a0proj/state.json)
        """
        if state_file is None:
            state_file = self.state_dir / "state.json"
        else:
            state_file = Path(state_file)

        state_data = {
            "project_path": str(self.project_path),
            "goal": self.goal,
            "context": self.context,
            "completed_steps": self.completed_steps,
            "current_step": self.current_step,
            "files_modified": self.files_modified,
            "decisions": self.decisions,
            "risks": self.risks,
            "checkpoints": self.checkpoints,
            "session_history": self.session_history,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_checkpoint": self.last_checkpoint.isoformat() if self.last_checkpoint else None
        }

        state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(state_file, "w") as f:
            json.dump(state_data, f, indent=2)

    def load_state(self, state_file: Optional[str] = None) -> None:
        """
        Load project state from JSON file.

        Args:
            state_file: Path to load state from (default: .a0proj/state.json)
        """
        if state_file is None:
            state_file = self.state_dir / "state.json"
        else:
            state_file = Path(state_file)

        with open(state_file, "r") as f:
            data = json.load(f)

        self.goal = data.get("goal")
        self.context = data.get("context", {})
        self.completed_steps = data.get("completed_steps", [])
        self.current_step = data.get("current_step")
        self.files_modified = data.get("files_modified", [])
        self.decisions = data.get("decisions", [])
        self.risks = data.get("risks", [])
        self.checkpoints = data.get("checkpoints", [])
        self.session_history = data.get("session_history", [])
        self.created_at = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now()
        self.updated_at = datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now()
        self.last_checkpoint = datetime.fromisoformat(data["last_checkpoint"]) if data.get("last_checkpoint") else None

    def is_paused_task(self) -> bool:
        """
        Check if this is a paused/resumed task.

        Returns:
            True if task has work remaining
        """
        progress = self.get_progress()
        return (
            progress["current_step"] is not None or
            len(self.get_remaining_work()) > 0
        )

    def can_resume(self) -> bool:
        """
        Check if work can be resumed.

        Returns:
            True if resume is possible
        """
        return self.is_paused_task() and self.goal is not None
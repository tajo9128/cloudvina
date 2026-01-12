import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from functools import wraps
from enum import Enum
import traceback

"""
Phase 1.4B: Execution Tracer

Purpose: Create a ground-truth timeline of actions
Agentic Impact: Enables rollback, replay, safe autonomy, Git governance
"""


class ActionStatus(Enum):
    """Status of an execution action."""
    STARTED = "started"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    ROLLED_BACK = "rolled_back"
    SKIPPED = "skipped"


class ActionType(Enum):
    """Types of actions an agent can execute."""
    TOOL_CALL = "tool_call"           # Using a tool (code execution, search, etc.)
    FILE_READ = "file_read"           # Reading a file
    FILE_WRITE = "file_write"         # Writing a file
    FILE_DELETE = "file_delete"       # Deleting a file
    CODE_EXECUTE = "code_execute"     # Executing code
    API_CALL = "api_call"             # Making an API call
    GIT_OPERATION = "git_operation"   # Git commands
    SUBORDINATE_CALL = "subordinate_call" # Calling another agent
    MEMORY_OPERATION = "memory_operation" # Memory read/write
    USER_INTERACTION = "user_interaction" # Waiting for user input
    SYSTEM_COMMAND = "system_command" # Terminal commands
    OTHER = "other"                   # Other actions


class ExecutionTracer:
    """
    Traces every action with full execution context.

    Traces:
    - Action type: What kind of action
    - Tool/command: Which tool was used
    - Inputs: What was passed in
    - Outputs: What was returned
    - Status: Success/failure/partial
    - Duration: How long it took
    - Side effects: What changed (files, memory, etc.)
    - Error info: Stack traces and error messages
    - Context: Additional context
    """

    def __init__(self, session_id: str, log_dir: Optional[str] = None):
        """
        Initialize ExecutionTracer for a session.

        Args:
            session_id: Unique identifier for the session
            log_dir: Directory to store logs (default: tmp/chats/<session_id>/)
        """
        self.session_id = session_id

        if log_dir:
            self.log_dir = Path(log_dir).resolve()
        else:
            self.log_dir = Path("/a0/tmp/chats") / session_id

        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Execution trace
        self.actions: List[Dict] = []
        self.session_start = datetime.now()
        self.current_action = None
        self.nested_stack = []

    def start_action(self,
                    action_type: ActionType,
                    tool: str,
                    inputs: Dict[str, Any],
                    context: Optional[Dict] = None) -> str:
        """
        Start tracing a new action.

        Args:
            action_type: Type of action
            tool: Tool or command being used
            inputs: Inputs passed to the tool
            context: Additional context

        Returns:
            Action ID
        """
        action_id = f"act_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:19]}"

        action_entry = {
            "id": action_id,
            "type": action_type.value,
            "tool": tool,
            "inputs": inputs,
            "outputs": None,
            "status": ActionStatus.STARTED.value,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "duration_seconds": None,
            "side_effects": [],
            "errors": [],
            "context": context or {},
            "parent_id": self.nested_stack[-1] if self.nested_stack else None,
            "children": []
        }

        self.actions.append(action_entry)
        self.current_action = action_id
        self.nested_stack.append(action_id)

        return action_id

    def end_action(self,
                  action_id: str,
                  status: ActionStatus,
                  outputs: Optional[Dict] = None,
                  side_effects: Optional[List[Dict]] = None,
                  errors: Optional[List[str]] = None) -> None:
        """
        End tracing an action.

        Args:
            action_id: Action ID to end
            status: Final status of the action
            outputs: Outputs from the action
            side_effects: List of side effects (file changes, memory updates)
            errors: List of errors if any
        """
        action = self.get_action(action_id)
        if not action:
            return

        end_time = datetime.now()
        start_time = datetime.fromisoformat(action["start_time"])
        duration = (end_time - start_time).total_seconds()

        action["status"] = status.value
        action["outputs"] = outputs
        action["end_time"] = end_time.isoformat()
        action["duration_seconds"] = round(duration, 3)
        action["side_effects"] = side_effects or []
        action["errors"] = errors or []

        # Pop from nested stack
        if self.nested_stack and self.nested_stack[-1] == action_id:
            self.nested_stack.pop()
            self.current_action = self.nested_stack[-1] if self.nested_stack else None

    def get_action(self, action_id: str) -> Optional[Dict]:
        """
        Get a specific action by ID.

        Args:
            action_id: Action ID to retrieve

        Returns:
            Action dict or None if not found
        """
        for action in self.actions:
            if action["id"] == action_id:
                return action
        return None

    def get_actions_by_type(self, action_type: ActionType) -> List[Dict]:
        """
        Get all actions of a specific type.

        Args:
            action_type: Type of action to retrieve

        Returns:
            List of actions
        """
        return [a for a in self.actions if a["type"] == action_type.value]

    def get_failed_actions(self) -> List[Dict]:
        """
        Get all failed actions.

        Returns:
            List of failed actions
        """
        return [a for a in self.actions if a["status"] == ActionStatus.FAILED.value]

    def get_successful_actions(self) -> List[Dict]:
        """
        Get all successful actions.

        Returns:
            List of successful actions
        """
        return [a for a in self.actions if a["status"] == ActionStatus.SUCCESS.value]

    def get_execution_timeline(self) -> List[Dict]:
        """
        Get actions in chronological order.

        Returns:
            List of actions with timing info
        """
        return sorted(self.actions, key=lambda x: x["start_time"])

    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the execution trace.

        Returns:
            Summary dict with statistics
        """
        type_counts = {}
        status_counts = {}
        total_duration = 0
        file_changes = 0

        for action in self.actions:
            # Count by type
            atype = action["type"]
            type_counts[atype] = type_counts.get(atype, 0) + 1

            # Count by status
            status = action["status"]
            status_counts[status] = status_counts.get(status, 0) + 1

            # Sum duration
            if action["duration_seconds"]:
                total_duration += action["duration_seconds"]

            # Count file changes
            for effect in action.get("side_effects", []):
                if effect.get("type") in ["file_write", "file_delete", "file_move"]:
                    file_changes += 1

        avg_duration = total_duration / len(self.actions) if self.actions else 0

        return {
            "total_actions": len(self.actions),
            "action_types": type_counts,
            "status_distribution": status_counts,
            "total_duration_seconds": round(total_duration, 3),
            "average_duration_seconds": round(avg_duration, 3),
            "file_changes": file_changes,
            "session_start": self.session_start.isoformat(),
            "session_duration_seconds": (datetime.now() - self.session_start).total_seconds()
        }

    def get_action_tree(self) -> List[Dict]:
        """
        Get actions in hierarchical tree form (parent-child relationships).

        Returns:
            Tree structure of actions
        """
        # Build a mapping of id -> action
        action_map = {a["id"]: a for a in self.actions}

        # Build children lists
        for action in self.actions:
            parent_id = action.get("parent_id")
            if parent_id and parent_id in action_map:
                action_map[parent_id]["children"].append(action["id"])

        # Get root actions (no parent)
        roots = [a for a in self.actions if a.get("parent_id") is None]

        return roots

    def trace_action(self, action_type: ActionType, tool: str):
        """
        Decorator to automatically trace function calls.

        Usage:
            @tracer.trace_action(ActionType.TOOL_CALL, "code_execution")
            def execute_code(code):
                ...
        """
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Start trace
                inputs = {
                    "args": [str(arg) for arg in args],
                    "kwargs": {k: str(v) for k, v in kwargs.items()}
                }
                action_id = self.start_action(action_type, tool, inputs)

                try:
                    # Execute function
                    result = func(*args, **kwargs)

                    # End trace with success
                    self.end_action(
                        action_id,
                        ActionStatus.SUCCESS,
                        outputs={"result": str(result) if result else None}
                    )

                    return result

                except Exception as e:
                    # End trace with failure
                    error_msg = str(e)
                    error_trace = traceback.format_exc()

                    self.end_action(
                        action_id,
                        ActionStatus.FAILED,
                        errors=[error_msg],
                        context={"traceback": error_trace}
                    )

                    raise

            return wrapper
        return decorator

    def generate_execution_report(self) -> str:
        """
        Generate a human-readable execution report.

        Returns:
            Markdown-formatted execution report
        """
        summary = self.get_execution_summary()
        failed = self.get_failed_actions()
        timeline = self.get_execution_timeline()

        lines = [
            "# Execution Trace Report",
            "",
            f"*Generated: {datetime.now().isoformat()}*",
            "",
            "## Summary",
            "",
            f"- **Total Actions:** {summary['total_actions']}",
            f"- **Total Duration:** {summary['total_duration_seconds']:.2f}s",
            f"- **Average Duration:** {summary['average_duration_seconds']:.3f}s",
            f"- **File Changes:** {summary['file_changes']}",
            "",
            "## Action Types",
            ""
        ]

        for atype, count in summary['action_types'].items():
            lines.append(f"- **{atype}:** {count}")

        lines.extend([
            "",
            "## Status Distribution",
            ""
        ])

        for status, count in summary['status_distribution'].items():
            lines.append(f"- **{status}:** {count}")

        if failed:
            lines.extend([
                "",
                "## Failed Actions",
                ""
            ])
            for action in failed:
                lines.append(f"### {action['tool']}")
                lines.append(f"- **Status:** {action['status']}")
                lines.append(f"- **Duration:** {action.get('duration_seconds', 'N/A')}s")
                if action.get('errors'):
                    lines.append(f"- **Errors:**")
                    for error in action['errors']:
                        lines.append(f"  - {error}")
                lines.append("")

        lines.extend([
            "",
            "## Execution Timeline",
            ""
        ])

        for action in timeline:
            duration = action.get("duration_seconds")
            duration_str = f"({duration:.3f}s)" if duration else "(running)"
            lines.append(f"- **{action['type']}** `{action['tool']}` {duration_str}")
            if action.get('errors'):
                lines.append(f"  ❌ {', '.join(action['errors'])}")

        return "\n".join(lines)

    def save_trace(self, trace_file: Optional[str] = None) -> None:
        """
        Save execution trace to JSON file.

        Args:
            trace_file: Path to save trace (default: tmp/chats/<session_id>/execution_trace.json)
        """
        if trace_file is None:
            trace_file = self.log_dir / "execution_trace.json"
        else:
            trace_file = Path(trace_file)

        trace_data = {
            "session_id": self.session_id,
            "session_start": self.session_start.isoformat(),
            "total_actions": len(self.actions),
            "actions": self.actions
        }

        with open(trace_file, "w") as f:
            json.dump(trace_data, f, indent=2)

    def load_trace(self, trace_file: Optional[str] = None) -> None:
        """
        Load execution trace from JSON file.

        Args:
            trace_file: Path to load trace from
        """
        if trace_file is None:
            trace_file = self.log_dir / "execution_trace.json"
        else:
            trace_file = Path(trace_file)

        with open(trace_file, "r") as f:
            data = json.load(f)

        self.session_id = data["session_id"]
        self.session_start = datetime.fromisoformat(data["session_start"])
        self.actions = data["actions"]

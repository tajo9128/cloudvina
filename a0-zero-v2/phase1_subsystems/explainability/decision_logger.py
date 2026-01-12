import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum

"""
Phase 1.4A: Decision Logger

Purpose: Capture reasoning, not just outcomes
Agentic Impact: Enables trust, audits, research validation, debugging agent behavior
"""


class DecisionType(Enum):
    """Types of decisions an agent can make."""
    PLAN = "plan"                      # Choosing a plan of action
    TOOL_SELECTION = "tool_selection"  # Selecting which tool to use
    MODEL_SELECTION = "model_selection" # Selecting which LLM model
    REFACTORING = "refactoring"        # How to refactor code
    ALGORITHM = "algorithm"            # Choosing an algorithm
    ARCHITECTURE = "architecture"      # Architectural decisions
    WORKFLOW = "workflow"              # Workflow decisions
    RISK_ACCEPTANCE = "risk_acceptance" # Deciding to accept a risk
    ROLLBACK = "rollback"              # Deciding to rollback changes
    PIVOT = "pivot"                    # Changing direction
    OTHER = "other"                    # Other decisions


class DecisionLogger:
    """
    Logs every decision with full reasoning context.

    Logs:
    - Intent: What was the agent trying to achieve
    - Decision: What was chosen
    - Reasoning: Why this decision was made
    - Alternatives: What other options were considered
    - Rejected: Why alternatives were rejected
    - Risk level: How risky is this decision
    - Model used: Which LLM made this decision
    - Confidence: How confident is the agent
    """

    def __init__(self, session_id: str, log_dir: Optional[str] = None):
        """
        Initialize DecisionLogger for a session.

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

        # Decision log
        self.decisions: List[Dict] = []
        self.session_start = datetime.now()
        self.last_decision = None

    def log_decision(self,
                    decision_type: DecisionType,
                    intent: str,
                    decision: str,
                    reasoning: str,
                    alternatives: Optional[List[str]] = None,
                    rejected_reasons: Optional[List[str]] = None,
                    risk_level: str = "medium",
                    model_used: Optional[str] = None,
                    confidence: float = 0.5,
                    context: Optional[Dict] = None) -> str:
        """
        Log a decision with full reasoning.

        Args:
            decision_type: Type of decision
            intent: What was the agent trying to achieve
            decision: What was chosen
            reasoning: Why this decision was made
            alternatives: What other options were considered
            rejected_reasons: Why alternatives were rejected
            risk_level: How risky (none, low, medium, high, critical)
            model_used: Which LLM made this decision
            confidence: Confidence level (0.0 to 1.0)
            context: Additional context

        Returns:
            Decision ID
        """
        decision_id = f"dec_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:19]}"

        decision_entry = {
            "id": decision_id,
            "type": decision_type.value,
            "intent": intent,
            "decision": decision,
            "reasoning": reasoning,
            "alternatives": alternatives or [],
            "rejected_reasons": rejected_reasons or [],
            "risk_level": risk_level,
            "model_used": model_used,
            "confidence": confidence,
            "context": context or {},
            "timestamp": datetime.now().isoformat(),
            "session_time_seconds": (datetime.now() - self.session_start).total_seconds()
        }

        self.decisions.append(decision_entry)
        self.last_decision = decision_id

        return decision_id

    def get_decision(self, decision_id: str) -> Optional[Dict]:
        """
        Get a specific decision by ID.

        Args:
            decision_id: Decision ID to retrieve

        Returns:
            Decision dict or None if not found
        """
        for decision in self.decisions:
            if decision["id"] == decision_id:
                return decision
        return None

    def get_decisions_by_type(self, decision_type: DecisionType) -> List[Dict]:
        """
        Get all decisions of a specific type.

        Args:
            decision_type: Type of decision to retrieve

        Returns:
            List of decisions
        """
        return [d for d in self.decisions if d["type"] == decision_type.value]

    def get_high_risk_decisions(self, min_risk: str = "high") -> List[Dict]:
        """
        Get all decisions at or above a risk level.

        Args:
            min_risk: Minimum risk level (none, low, medium, high, critical)

        Returns:
            List of high-risk decisions
        """
        risk_order = ["none", "low", "medium", "high", "critical"]
        min_risk_index = risk_order.index(min_risk)

        return [
            d for d in self.decisions
            if risk_order.index(d["risk_level"]) >= min_risk_index
        ]

    def get_low_confidence_decisions(self, max_confidence: float = 0.3) -> List[Dict]:
        """
        Get decisions with low confidence.

        Args:
            max_confidence: Maximum confidence threshold

        Returns:
            List of low-confidence decisions
        """
        return [
            d for d in self.decisions
            if d["confidence"] <= max_confidence
        ]

    def get_decision_timeline(self) -> List[Dict]:
        """
        Get decisions in chronological order.

        Returns:
            List of decisions with timing info
        """
        return sorted(self.decisions, key=lambda x: x["timestamp"])

    def get_decision_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all decisions.

        Returns:
            Summary dict with statistics
        """
        type_counts = {}
        risk_counts = {}
        total_confidence = 0

        for decision in self.decisions:
            # Count by type
            dtype = decision["type"]
            type_counts[dtype] = type_counts.get(dtype, 0) + 1

            # Count by risk
            risk = decision["risk_level"]
            risk_counts[risk] = risk_counts.get(risk, 0) + 1

            # Sum confidence
            total_confidence += decision["confidence"]

        avg_confidence = total_confidence / len(self.decisions) if self.decisions else 0

        return {
            "total_decisions": len(self.decisions),
            "decision_types": type_counts,
            "risk_distribution": risk_counts,
            "average_confidence": round(avg_confidence, 3),
            "session_start": self.session_start.isoformat(),
            "session_duration_seconds": (datetime.now() - self.session_start).total_seconds(),
            "decisions_per_minute": round(len(self.decisions) / max(1, (datetime.now() - self.session_start).total_seconds() / 60), 2)
        }

    def explain_decision(self, decision_id: str) -> str:
        """
        Generate a human-readable explanation of a decision.

        Args:
            decision_id: Decision ID to explain

        Returns:
            Human-readable explanation
        """
        decision = self.get_decision(decision_id)
        if not decision:
            return f"Decision {decision_id} not found"

        lines = [
            f"# Decision: {decision['decision']}",
            f"",
            f"**Intent:** {decision['intent']}",
            f"",
            f"**Reasoning:** {decision['reasoning']}",
            f"",
            f"**Alternatives Considered:**"
        ]

        for i, alt in enumerate(decision['alternatives'], 1):
            rejected = decision['rejected_reasons'][i-1] if i <= len(decision['rejected_reasons']) else "No reason recorded"
            lines.append(f"  {i}. {alt} - Rejected: {rejected}")

        lines.extend([
            f"",
            f"**Risk Level:** {decision['risk_level']}",
            f"**Confidence:** {decision['confidence']}",
            f"**Model:** {decision.get('model_used', 'N/A')}",
            f"**Time:** {decision['timestamp']}"
        ])

        return "\\n".join(lines)

    def generate_audit_report(self) -> str:
        """
        Generate an audit report of all decisions.

        Returns:
            Markdown-formatted audit report
        """
        summary = self.get_decision_summary()
        high_risk = self.get_high_risk_decisions()
        low_conf = self.get_low_confidence_decisions()

        lines = [
            "# Decision Audit Report",
            f"",
            f"*Generated: {datetime.now().isoformat()}*",
            f"",
            f"## Summary",
            f"",
            f"- **Total Decisions:** {summary['total_decisions']}",
            f"- **Average Confidence:** {summary['average_confidence']}",
            f"- **Session Duration:** {summary['session_duration_seconds']:.1f}s",
            f"- **Decisions/Minute:** {summary['decisions_per_minute']}",
            f"",
            f"## Decision Types",
            f""
        ]

        for dtype, count in summary['decision_types'].items():
            lines.append(f"- **{dtype}:** {count}")

        lines.extend([
            f"",
            f"## Risk Distribution",
            f""
        ])

        for risk, count in summary['risk_distribution'].items():
            lines.append(f"- **{risk}:** {count}")

        if high_risk:
            lines.extend([
                f"",
                f"## High-Risk Decisions",
                f""
            ])
            for decision in high_risk:
                lines.append(f"### {decision['decision']}")
                lines.append(f"- **Intent:** {decision['intent']}")
                lines.append(f"- **Reasoning:** {decision['reasoning']}")
                lines.append(f"- **Confidence:** {decision['confidence']}")
                lines.append(f"")

        if low_conf:
            lines.extend([
                f"",
                f"## Low-Confidence Decisions",
                f""
            ])
            for decision in low_conf:
                lines.append(f"### {decision['decision']}")
                lines.append(f"- **Intent:** {decision['intent']}")
                lines.append(f"- **Reasoning:** {decision['reasoning']}")
                lines.append(f"- **Confidence:** {decision['confidence']}")
                lines.append(f"")

        return "\\n".join(lines)

    def save_log(self, log_file: Optional[str] = None) -> None:
        """
        Save decision log to JSON file.

        Args:
            log_file: Path to save log (default: tmp/chats/<session_id>/decisions.json)
        """
        if log_file is None:
            log_file = self.log_dir / "decisions.json"
        else:
            log_file = Path(log_file)

        log_data = {
            "session_id": self.session_id,
            "session_start": self.session_start.isoformat(),
            "total_decisions": len(self.decisions),
            "decisions": self.decisions
        }

        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2)

    def load_log(self, log_file: Optional[str] = None) -> None:
        """
        Load decision log from JSON file.

        Args:
            log_file: Path to load log from
        """
        if log_file is None:
            log_file = self.log_dir / "decisions.json"
        else:
            log_file = Path(log_file)

        with open(log_file, "r") as f:
            data = json.load(f)

        self.session_id = data["session_id"]
        self.session_start = datetime.fromisoformat(data["session_start"])
        self.decisions = data["decisions"]

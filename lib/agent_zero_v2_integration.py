
"""
Agent Zero v2 Integration Module

This module integrates all IDE-grade improvements:
- Phase 1: Repo Awareness, Project State, Explainability
- Phase 2: Git Governor, Model Router
- Speed Optimizations: Phase 1 & Phase 2

Activates all improvements automatically in all chats.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Add lib directory to path
LIB_PATH = Path(__file__).parent
if LIB_PATH.exists():
    sys.path.insert(0, str(LIB_PATH))

# Import Phase 1 Subsystems
try:
    from repo_awareness.file_indexer import FileIndexer
    from repo_awareness.dependency_graph import DependencyGraph
    from project_state.state_manager import ProjectStateManager
    from explainability.decision_logger import DecisionLogger
    from explainability.execution_tracer import ExecutionTracer
    from phase1_schemas.interfaces import IFileIndexer, IDependencyGraph
    PHASE1_AVAILABLE = True
except ImportError as e:
    PHASE1_AVAILABLE = False
    print(f"Phase 1 modules not available: {e}")

# Import Phase 2 Subsystems
try:
    from phase2_subsystems.git_governor.approval_workflow import ApprovalWorkflow
    from phase2_subsystems.git_governor.commit_validator import CommitValidator
    from phase2_subsystems.model_router.task_classifier import TaskClassifier
    from phase2_subsystems.model_router.model_selector import ModelSelector
    PHASE2_AVAILABLE = True
except ImportError as e:
    PHASE2_AVAILABLE = False
    print(f"Phase 2 modules not available: {e}")

# Import Speed Optimizations - Phase 1
try:
    from phase1_optimizations.cache_manager import CacheManager
    from phase1_optimizations.async_logger import AsyncLogger
    from phase1_optimizations.state_streaming import StateStreamer
    from phase1_optimizations.async_operations import AsyncOperations
    PHASE1_OPT_AVAILABLE = True
except ImportError as e:
    PHASE1_OPT_AVAILABLE = False
    print(f"Phase 1 optimizations not available: {e}")

# Import Speed Optimizations - Phase 2
try:
    from phase2_optimizations.planning_envelope import PlanningEnvelope
    from phase2_optimizations.fast_deep_path import FastDeepPathExecutor
    from phase2_optimizations.smart_llm_usage import DeterministicTaskDetector
    from phase2_optimizations.connection_pool import OptimizedLLMClient
    PHASE2_OPT_AVAILABLE = True
except ImportError as e:
    PHASE2_OPT_AVAILABLE = False
    print(f"Phase 2 optimizations not available: {e}")


class AgentZeroV2Integration:
    """
    Main integration class that activates all Agent Zero v2 improvements.
    """

    def __init__(self, project_path: Optional[str] = None):
        self.project_path = Path(project_path) if project_path else Path.cwd()
        self.initialized = False

        # Phase 1 Subsystems
        self.file_indexer: Optional[FileIndexer] = None
        self.dependency_graph: Optional[DependencyGraph] = None
        self.project_state: Optional[ProjectStateManager] = None
        self.decision_logger: Optional[DecisionLogger] = None
        self.execution_tracer: Optional[ExecutionTracer] = None

        # Phase 2 Subsystems
        self.approval_workflow: Optional[ApprovalWorkflow] = None
        self.commit_validator: Optional[CommitValidator] = None
        self.task_classifier: Optional[TaskClassifier] = None
        self.model_selector: Optional[ModelSelector] = None

        # Speed Optimizations - Phase 1
        self.cache_manager: Optional[CacheManager] = None
        self.async_logger: Optional[AsyncLogger] = None
        self.state_streamer: Optional[StateStreamer] = None
        self.async_ops: Optional[AsyncOperations] = None

        # Speed Optimizations - Phase 2
        self.planning_envelope: Optional[PlanningEnvelope] = None
        self.fast_deep_executor: Optional[FastDeepPathExecutor] = None
        self.deterministic_detector: Optional[DeterministicTaskDetector] = None
        self.llm_client: Optional[OptimizedLLMClient] = None

    def initialize(self):
        """Initialize all available subsystems."""
        print("[Agent Zero v2] Initializing integration...")

        # Initialize Speed Optimizations first (they help other subsystems)
        self._initialize_optimizations()

        # Initialize Phase 1 Subsystems
        if PHASE1_AVAILABLE:
            self._initialize_phase1()

        # Initialize Phase 2 Subsystems
        if PHASE2_AVAILABLE:
            self._initialize_phase2()

        self.initialized = True
        print("[Agent Zero v2] Integration complete!")

        return self.get_status()

    def _initialize_optimizations(self):
        """Initialize speed optimization modules."""
        print("[Agent Zero v2] Initializing Speed Optimizations...")

        # Phase 1 Optimizations
        if PHASE1_OPT_AVAILABLE:
            try:
                self.cache_manager = CacheManager(ttl_seconds=3600)
                self.async_logger = AsyncLogger()
                self.state_streamer = StateStreamer()
                self.async_ops = AsyncOperations(max_workers=4)
                print("[Agent Zero v2] ✓ Phase 1 Optimizations loaded")
            except Exception as e:
                print(f"[Agent Zero v2] ✗ Phase 1 Optimizations failed: {e}")

        # Phase 2 Optimizations
        if PHASE2_OPT_AVAILABLE:
            try:
                self.planning_envelope = PlanningEnvelope()
                self.fast_deep_executor = FastDeepPathExecutor()
                self.deterministic_detector = DeterministicTaskDetector()
                self.llm_client = OptimizedLLMClient()
                print("[Agent Zero v2] ✓ Phase 2 Optimizations loaded")
            except Exception as e:
                print(f"[Agent Zero v2] ✗ Phase 2 Optimizations failed: {e}")

    def _initialize_phase1(self):
        """Initialize Phase 1 subsystems."""
        print("[Agent Zero v2] Initializing Phase 1 Subsystems...")

        try:
            # Repo Awareness
            self.file_indexer = FileIndexer(str(self.project_path))
            self.file_indexer.index()

            self.dependency_graph = DependencyGraph(str(self.project_path))
            self.dependency_graph.build()

            print("[Agent Zero v2] ✓ Repo Awareness loaded")
        except Exception as e:
            print(f"[Agent Zero v2] ✗ Repo Awareness failed: {e}")

        try:
            # Project State
            self.project_state = ProjectStateManager(str(self.project_path))
            self.project_state.initialize()

            print("[Agent Zero v2] ✓ Project State loaded")
        except Exception as e:
            print(f"[Agent Zero v2] ✗ Project State failed: {e}")

        try:
            # Explainability
            self.decision_logger = DecisionLogger()
            self.execution_tracer = ExecutionTracer()

            print("[Agent Zero v2] ✓ Explainability loaded")
        except Exception as e:
            print(f"[Agent Zero v2] ✗ Explainability failed: {e}")

    def _initialize_phase2(self):
        """Initialize Phase 2 subsystems."""
        print("[Agent Zero v2] Initializing Phase 2 Subsystems...")

        try:
            # Git Governor
            self.approval_workflow = ApprovalWorkflow()
            self.commit_validator = CommitValidator()

            print("[Agent Zero v2] ✓ Git Governor loaded")
        except Exception as e:
            print(f"[Agent Zero v2] ✗ Git Governor failed: {e}")

        try:
            # Model Router
            self.task_classifier = TaskClassifier()
            self.model_selector = ModelSelector()

            print("[Agent Zero v2] ✓ Model Router loaded")
        except Exception as e:
            print(f"[Agent Zero v2] ✗ Model Router failed: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current status of all subsystems."""
        return {
            "integration_version": "2.0",
            "initialized": self.initialized,
            "project_path": str(self.project_path),
            "phase1_subsystems": {
                "repo_awareness": bool(self.file_indexer and self.dependency_graph),
                "project_state": bool(self.project_state),
                "explainability": bool(self.decision_logger and self.execution_tracer)
            },
            "phase2_subsystems": {
                "git_governor": bool(self.approval_workflow and self.commit_validator),
                "model_router": bool(self.task_classifier and self.model_selector)
            },
            "phase1_optimizations": {
                "cache_manager": bool(self.cache_manager),
                "async_logger": bool(self.async_logger),
                "state_streamer": bool(self.state_streamer),
                "async_operations": bool(self.async_ops)
            },
            "phase2_optimizations": {
                "planning_envelope": bool(self.planning_envelope),
                "fast_deep_path": bool(self.fast_deep_executor),
                "deterministic_detector": bool(self.deterministic_detector),
                "optimized_llm_client": bool(self.llm_client)
            },
            "timestamp": datetime.now().isoformat()
        }

    def log_decision(self, decision: str, reasoning: List[str], alternatives: List[str]):
        """Log a decision with reasoning and alternatives."""
        if self.decision_logger:
            self.decision_logger.log(
                decision=decision,
                reasoning=reasoning,
                alternatives=alternatives,
                confidence=0.8
            )

    def trace_execution(self, action: str, input_data: Any, output_data: Any):
        """Trace an execution action."""
        if self.execution_tracer:
            self.execution_tracer.trace(
                action=action,
                input_data=str(input_data)[:200],
                output_data=str(output_data)[:200],
                status="success"
            )

    def classify_task(self, task_description: str) -> str:
        """Classify a task for routing."""
        if self.task_classifier:
            return self.task_classifier.classify(task_description)
        return "default"

    def select_model(self, task_type: str) -> str:
        """Select appropriate model for task type."""
        if self.model_selector:
            model = self.model_selector.select(task_type)
            return model.get("model_id", "default")
        return "default"


# Singleton instance
_integration_instance: Optional[AgentZeroV2Integration] = None


def get_integration(project_path: Optional[str] = None) -> AgentZeroV2Integration:
    """Get or create the integration instance."""
    global _integration_instance

    if _integration_instance is None:
        _integration_instance = AgentZeroV2Integration(project_path)
        _integration_instance.initialize()

    return _integration_instance

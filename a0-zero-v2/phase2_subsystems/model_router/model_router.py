"""
Model Router - Main integration for intelligent LLM routing.
Combines task classification, model selection, and cost optimization.
"""
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid

import sys
sys.path.insert(0, '/a0/lib')

from phase2_subsystems.model_router.interfaces import (
    IModelRouter, TaskCategory, TaskClassification, ModelSelection,
    RoutingDecision, ModelMetadata, UsageMetrics
)
from phase2_subsystems.model_router.task_classifier import (
    TaskClassifier, TaskClassifierConfig
)
from phase2_subsystems.model_router.model_selector import (
    ModelSelector, ModelSelectorConfig
)
from phase2_subsystems.model_router.cost_optimizer import (
    CostOptimizer, CostOptimizerConfig
)


class ModelRouterConfig:
    """Configuration for ModelRouter."""
    def __init__(
        self,
        classifier_config: Optional[TaskClassifierConfig] = None,
        selector_config: Optional[ModelSelectorConfig] = None,
        optimizer_config: Optional[CostOptimizerConfig] = None,
        auto_track: bool = True,
        auto_optimize: bool = True
    ):
        self.classifier_config = classifier_config or TaskClassifierConfig()
        self.selector_config = selector_config or ModelSelectorConfig()
        self.optimizer_config = optimizer_config or CostOptimizerConfig()
        self.auto_track = auto_track
        self.auto_optimize = auto_optimize


class ModelRouter(IModelRouter):
    """Main Model Router for intelligent LLM routing."""
    
    def __init__(self, config: Optional[ModelRouterConfig] = None):
        self.config = config or ModelRouterConfig()
        
        # Initialize components
        self.classifier = TaskClassifier(self.config.classifier_config)
        self.selector = ModelSelector(self.config.selector_config)
        self.optimizer = CostOptimizer(self.config.optimizer_config)
        
        # Routing history
        self._routing_history: List[RoutingDecision] = []
    
    def route(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> RoutingDecision:
        """Route task to optimal model."""
        # Generate task ID
        task_id = str(uuid.uuid4())[:8]
        
        # Classify task
        classification = self.classifier.classify(task_description, context)
        
        # Check for optimized model
        if self.config.auto_optimize:
            optimized_model = self.optimizer.optimize_selection(classification)
            if optimized_model:
                # Override selection with optimized model
                selection = ModelSelection(
                    model=optimized_model,
                    reason=f"Optimized choice for {classification.category.value} based on usage",
                    expected_cost=(classification.estimated_tokens / 1000) * optimized_model.cost_per_1k_tokens,
                    expected_latency_ms=optimized_model.average_latency_ms,
                    confidence=0.9
                )
            else:
                # Use regular selection
                selection = self.selector.select_model(classification)
        else:
            selection = self.selector.select_model(classification)
        
        # Create routing decision
        decision = RoutingDecision(
            task_id=task_id,
            classification=classification,
            selection=selection,
            timestamp=datetime.now().isoformat(),
            status="pending"
        )
        
        # Store in history
        self._routing_history.append(decision)
        
        return decision
    
    def classify_task(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> TaskClassification:
        """Classify task without routing."""
        return self.classifier.classify(task_description, context)
    
    def select_model(self, classification: TaskCategory) -> ModelSelection:
        """Select model for given category."""
        # Create a dummy classification for the selector
        from phase2_subsystems.model_router.interfaces import TaskClassification as TC
        dummy_classification = TC(
            category=classification,
            confidence=1.0,
            reason="Direct category selection"
        )
        return self.selector.select_model(dummy_classification)
    
    def get_usage_metrics(self) -> UsageMetrics:
        """Get usage metrics."""
        return self.optimizer.get_metrics()
    
    def get_available_models(self) -> List[ModelMetadata]:
        """Get available models."""
        return self.selector.get_available_models()
    
    def add_model(self, model: ModelMetadata) -> bool:
        """Add a model to the pool."""
        return self.selector.add_model(model)
    
    def remove_model(self, model_name: str) -> bool:
        """Remove a model from the pool."""
        return self.selector.remove_model(model_name)
    
    def track_execution(self, decision: RoutingDecision, actual_tokens: int, latency_ms: float):
        """Track execution metrics."""
        if self.config.auto_track:
            decision.status = "completed"
            self.optimizer.track_usage(decision, actual_tokens, latency_ms)
    
    def mark_failed(self, decision: RoutingDecision, error: str):
        """Mark decision as failed."""
        decision.status = "failed"
        decision.error = error
    
    def get_routing_history(self, limit: int = 50) -> List[RoutingDecision]:
        """Get recent routing decisions."""
        return self._routing_history[-limit:]
    
    def get_cost_report(self) -> str:
        """Generate cost report."""
        return self.optimizer.get_cost_report()
    
    def reset_metrics(self):
        """Reset all metrics."""
        self.optimizer.reset_metrics()
        self._routing_history.clear()


# Convenience function
def create_router() -> ModelRouter:
    """Create a configured ModelRouter instance."""
    return ModelRouter()

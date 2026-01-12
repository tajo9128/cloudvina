"""
Model Selector - Selects optimal models for tasks.
Uses category-based mapping with cost and quality considerations.
"""
from typing import List, Optional

import sys
sys.path.insert(0, '/a0/lib')

from phase2_subsystems.model_router.interfaces import (
    IModelSelector, ModelMetadata, TaskCategory, ModelCapability,
    ModelProvider, ModelSelection
)


class ModelSelectorConfig:
    """Configuration for ModelSelector."""
    def __init__(
        self,
        quality_weight: float = 0.5,
        cost_weight: float = 0.3,
        latency_weight: float = 0.2,
        fallback_enabled: bool = True
    ):
        self.quality_weight = quality_weight
        self.cost_weight = cost_weight
        self.latency_weight = latency_weight
        self.fallback_enabled = fallback_enabled


class ModelSelector(IModelSelector):
    """Intelligent model selector for task routing."""
    
    # Default model pool
    DEFAULT_MODELS = [
        # High quality models
        ModelMetadata(
            name="gpt-4",
            provider=ModelProvider.OPENAI,
            capability=ModelCapability.HIGH_QUALITY,
            context_length=8192,
            cost_per_1k_tokens=0.03,
            average_latency_ms=2000,
            quality_score=0.98,
            reliability_score=0.99
        ),
        ModelMetadata(
            name="claude-3-opus",
            provider=ModelProvider.ANTHROPIC,
            capability=ModelCapability.HIGH_QUALITY,
            context_length=200000,
            cost_per_1k_tokens=0.015,
            average_latency_ms=2500,
            quality_score=0.97,
            reliability_score=0.98
        ),
        # Code specialized models
        ModelMetadata(
            name="glm-4.7",
            provider=ModelProvider.GLM,
            capability=ModelCapability.CODE_SPECIALIZED,
            context_length=128000,
            cost_per_1k_tokens=0.002,
            average_latency_ms=1500,
            quality_score=0.85,
            reliability_score=0.95
        ),
        ModelMetadata(
            name="gpt-4-turbo",
            provider=ModelProvider.OPENAI,
            capability=ModelCapability.CODE_SPECIALIZED,
            context_length=128000,
            cost_per_1k_tokens=0.01,
            average_latency_ms=1000,
            quality_score=0.92,
            reliability_score=0.99
        ),
        # Fast/cheap models
        ModelMetadata(
            name="mistral:latest",
            provider=ModelProvider.OLLAMA,
            capability=ModelCapability.FAST_CHEAP,
            context_length=32768,
            cost_per_1k_tokens=0.0,
            average_latency_ms=500,
            quality_score=0.75,
            reliability_score=0.90,
            max_concurrent=5
        ),
        ModelMetadata(
            name="gpt-3.5-turbo",
            provider=ModelProvider.OPENAI,
            capability=ModelCapability.FAST_CHEAP,
            context_length=16385,
            cost_per_1k_tokens=0.001,
            average_latency_ms=400,
            quality_score=0.80,
            reliability_score=0.98,
            max_concurrent=10
        ),
    ]
    
    # Category capability mapping
    CATEGORY_CAPABILITIES = {
        TaskCategory.PLANNING: ModelCapability.HIGH_QUALITY,
        TaskCategory.REFACTOR: ModelCapability.HIGH_QUALITY,
        TaskCategory.CODE_GEN: ModelCapability.CODE_SPECIALIZED,
        TaskCategory.DEBUG: ModelCapability.CODE_SPECIALIZED,
        TaskCategory.TEST: ModelCapability.CODE_SPECIALIZED,
        TaskCategory.DOCS: ModelCapability.FAST_CHEAP,
        TaskCategory.QUERY: ModelCapability.FAST_CHEAP,
    }
    
    def __init__(self, config: Optional[ModelSelectorConfig] = None):
        self.config = config or ModelSelectorConfig()
        self._models = self.DEFAULT_MODELS.copy()
    
    def select_model(self, classification) -> ModelSelection:
        """Select optimal model for task."""
        # Import interfaces here to avoid circular imports
        from phase2_subsystems.model_router.interfaces import TaskClassification
        
        if not isinstance(classification, TaskClassification):
            # Handle if task category is passed directly
            classification = TaskClassification(
                category=classification,
                confidence=0.9,
                reason="Direct category selection"
            )
        
        # Get required capability
        required_capability = self.CATEGORY_CAPABILITIES.get(
            classification.category,
            ModelCapability.FAST_CHEAP
        )
        
        # Filter models by capability
        capable_models = [
            model for model in self._models
            if model.capability == required_capability
        ]
        
        # If vision required, filter further
        if classification.requires_vision:
            capable_models = [
                model for model in capable_models
                if model.supports_vision
            ]
        
        # If no capable models, use all available
        if not capable_models and self.config.fallback_enabled:
            capable_models = self._models
        
        # Score models
        scored_models = []
        for model in capable_models:
            score = self._score_model(model, classification)
            scored_models.append((score, model))
        
        # Sort by score
        scored_models.sort(key=lambda x: x[0], reverse=True)
        
        # Get best model
        if scored_models:
            best_score, best_model = scored_models[0]
            alternatives = [model for _, model in scored_models[1:3]]
        else:
            # Fallback to first available
            best_model = self._models[0] if self._models else None
            best_score = 0.0
            alternatives = self._models[1:3]
        
        # Calculate expected cost and latency
        expected_tokens = classification.estimated_tokens
        expected_cost = (expected_tokens / 1000) * best_model.cost_per_1k_tokens if best_model else 0
        expected_latency = best_model.average_latency_ms if best_model else 0
        
        # Generate reason
        reason = (
            f"Selected {best_model.name} for {classification.category.value} "
            f"with {best_model.capability.value} capability "
            f"(score: {best_score:.2f})"
        )
        
        return ModelSelection(
            model=best_model,
            reason=reason,
            expected_cost=expected_cost,
            expected_latency_ms=expected_latency,
            confidence=min(best_score, 1.0),
            alternatives=alternatives
        )
    
    def get_available_models(self) -> List[ModelMetadata]:
        """Get list of available models."""
        return self._models.copy()
    
    def add_model(self, model: ModelMetadata) -> bool:
        """Add a model to available pool."""
        if not isinstance(model, ModelMetadata):
            return False
        
        # Check if model already exists
        if any(m.name == model.name for m in self._models):
            return False
        
        self._models.append(model)
        return True
    
    def remove_model(self, model_name: str) -> bool:
        """Remove a model from available pool."""
        original_count = len(self._models)
        self._models = [m for m in self._models if m.name != model_name]
        return len(self._models) < original_count
    
    def _score_model(self, model: ModelMetadata, classification) -> float:
        """Score a model based on task requirements."""
        # Base score from quality
        score = model.quality_score * self.config.quality_weight
        
        # Add reliability component
        score += model.reliability_score * 0.1
        
        # Cost optimization (lower cost = higher score)
        if model.cost_per_1k_tokens > 0:
            cost_score = 1.0 / (model.cost_per_1k_tokens * 100)
            score += cost_score * self.config.cost_weight
        else:
            score += self.config.cost_weight
        
        # Latency optimization (lower latency = higher score)
        latency_score = 1.0 / (model.average_latency_ms / 1000)
        score += latency_score * self.config.latency_weight
        
        # Adjust for estimated tokens
        if classification.estimated_tokens > 5000:
            # Prefer models with larger context
            context_score = min(model.context_length / 100000, 1.0)
            score += context_score * 0.1
        
        return min(score, 1.0)

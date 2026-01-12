"""
Model Router - Interfaces for intelligent LLM routing.
Selects optimal models based on task classification, cost, and quality.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Literal
from dataclasses import dataclass, field
from enum import Enum


class TaskCategory(Enum):
    """Categories of tasks for model routing."""
    PLANNING = "planning"
    CODE_GEN = "code_gen"
    REFACTOR = "refactor"
    DEBUG = "debug"
    TEST = "test"
    DOCS = "docs"
    QUERY = "query"


class ModelCapability(Enum):
    """Model capability levels."""
    HIGH_QUALITY = "high_quality"  # Best for planning, refactoring
    CODE_SPECIALIZED = "code_specialized"  # Best for code generation, debugging
    FAST_CHEAP = "fast_cheap"  # Best for simple queries
    VISION = "vision"  # Multimodal capability


class ModelProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GLM = "glm"
    OLLAMA = "ollama"
    LOCAL = "local"
    CUSTOM = "custom"


@dataclass
class ModelMetadata:
    """Metadata for an LLM model."""
    name: str
    provider: ModelProvider
    capability: ModelCapability
    context_length: int
    cost_per_1k_tokens: float
    average_latency_ms: float
    supports_vision: bool = False
    quality_score: float = 0.9
    reliability_score: float = 0.9
    max_concurrent: int = 1


@dataclass
class TaskClassification:
    """Classification result for a task."""
    category: TaskCategory
    confidence: float
    reason: str
    alternative_categories: List[TaskCategory] = field(default_factory=list)
    estimated_tokens: int = 1000
    requires_vision: bool = False
    requires_code: bool = False


@dataclass
class ModelSelection:
    """Selected model for a task."""
    model: ModelMetadata
    reason: str
    expected_cost: float
    expected_latency_ms: float
    confidence: float
    alternatives: List[ModelMetadata] = field(default_factory=list)


@dataclass
class UsageMetrics:
    """Usage metrics for cost optimization."""
    total_tokens: int = 0
    total_cost: float = 0.0
    total_requests: int = 0
    model_usage: Dict[str, int] = field(default_factory=dict)
    category_usage: Dict[str, int] = field(default_factory=dict)
    average_latency_ms: float = 0.0
    success_rate: float = 1.0


@dataclass
class RoutingDecision:
    """Complete routing decision."""
    task_id: str
    classification: TaskClassification
    selection: ModelSelection
    timestamp: str = ""
    status: Literal["pending", "completed", "failed"] = "pending"
    error: Optional[str] = None


class ITaskClassifier(ABC):
    """Task classification interface."""
    
    @abstractmethod
    def classify(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> TaskClassification:
        """Classify a task into a category."""
        pass
    
    @abstractmethod
    def estimate_tokens(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> int:
        """Estimate token usage for task."""
        pass


class IModelSelector(ABC):
    """Model selection interface."""
    
    @abstractmethod
    def select_model(self, classification: TaskClassification) -> ModelSelection:
        """Select optimal model for task."""
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[ModelMetadata]:
        """Get list of available models."""
        pass
    
    @abstractmethod
    def add_model(self, model: ModelMetadata) -> bool:
        """Add a model to available pool."""
        pass
    
    @abstractmethod
    def remove_model(self, model_name: str) -> bool:
        """Remove a model from available pool."""
        pass


class ICostOptimizer(ABC):
    """Cost optimization interface."""
    
    @abstractmethod
    def track_usage(self, decision: RoutingDecision, actual_tokens: int, latency_ms: float):
        """Track usage metrics."""
        pass
    
    @abstractmethod
    def get_metrics(self) -> UsageMetrics:
        """Get current usage metrics."""
        pass
    
    @abstractmethod
    def optimize_selection(self, classification: TaskClassification) -> Optional[ModelMetadata]:
        """Suggest cost-optimized model."""
        pass
    
    @abstractmethod
    def reset_metrics(self):
        """Reset usage metrics."""
        pass


class IModelRouter(ABC):
    """Main Model Router interface."""
    
    @abstractmethod
    def route(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> RoutingDecision:
        """Route task to optimal model."""
        pass
    
    @abstractmethod
    def classify_task(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> TaskClassification:
        """Classify task without routing."""
        pass
    
    @abstractmethod
    def select_model(self, classification: TaskCategory) -> ModelSelection:
        """Select model for given category."""
        pass
    
    @abstractmethod
    def get_usage_metrics(self) -> UsageMetrics:
        """Get usage metrics."""
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[ModelMetadata]:
        """Get available models."""
        pass

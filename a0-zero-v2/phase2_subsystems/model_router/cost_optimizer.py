"""
Cost Optimizer - Tracks usage metrics and optimizes cost.
Monitors token usage, costs, and provides optimization suggestions.
"""
from typing import Optional, Dict, Any
from datetime import datetime
from collections import defaultdict

import sys
sys.path.insert(0, '/a0/lib')

from phase2_subsystems.model_router.interfaces import (
    ICostOptimizer, UsageMetrics, TaskClassification, ModelMetadata
)


class CostOptimizerConfig:
    """Configuration for CostOptimizer."""
    def __init__(
        self,
        cost_alert_threshold: float = 10.0,
        optimize_after_requests: int = 10,
        max_cost_per_session: float = 50.0
    ):
        self.cost_alert_threshold = cost_alert_threshold
        self.optimize_after_requests = optimize_after_requests
        self.max_cost_per_session = max_cost_per_session


class CostOptimizer(ICostOptimizer):
    """Tracks usage and provides cost optimization."""
    
    def __init__(self, config: Optional[CostOptimizerConfig] = None):
        self.config = config or CostOptimizerConfig()
        
        # Usage tracking
        self._metrics = UsageMetrics()
        
        # Request history for optimization
        self._request_history: list = []
        
        # Category model mapping (optimized choices)
        self._category_model_map: Dict[str, str] = {}
    
    def track_usage(self, decision: Any, actual_tokens: int, latency_ms: float):
        """Track usage metrics."""
        # Import interfaces here to avoid circular imports
        from phase2_subsystems.model_router.interfaces import RoutingDecision
        
        if not isinstance(decision, RoutingDecision):
            return
        
        # Update metrics
        self._metrics.total_tokens += actual_tokens
        self._metrics.total_requests += 1
        
        # Calculate cost
        cost = (actual_tokens / 1000) * decision.selection.model.cost_per_1k_tokens
        self._metrics.total_cost += cost
        
        # Update model usage
        model_name = decision.selection.model.name
        self._metrics.model_usage[model_name] = (
            self._metrics.model_usage.get(model_name, 0) + 1
        )
        
        # Update category usage
        category = decision.classification.category.value
        self._metrics.category_usage[category] = (
            self._metrics.category_usage.get(category, 0) + 1
        )
        
        # Update average latency
        total_latency = self._metrics.average_latency_ms * (self._metrics.total_requests - 1)
        self._metrics.average_latency_ms = (total_latency + latency_ms) / self._metrics.total_requests
        
        # Store request history
        self._request_history.append({
            'timestamp': datetime.now().isoformat(),
            'task_id': decision.task_id,
            'category': category,
            'model': model_name,
            'tokens': actual_tokens,
            'cost': cost,
            'latency_ms': latency_ms
        })
        
        # Check for cost alert
        if self._metrics.total_cost >= self.config.cost_alert_threshold:
            print(f"[CostOptimizer] Alert: Total cost ${self._metrics.total_cost:.2f} exceeded threshold ${self.config.cost_alert_threshold:.2f}")
    
    def get_metrics(self) -> UsageMetrics:
        """Get current usage metrics."""
        return self._metrics
    
    def optimize_selection(self, classification: TaskClassification) -> Optional[ModelMetadata]:
        """Suggest cost-optimized model."""
        # Check if we have enough data to optimize
        if self._metrics.total_requests < self.config.optimize_after_requests:
            return None
        
        # Get category
        category = classification.category.value
        
        # Check if we have optimized choice for this category
        if category in self._category_model_map:
            # Find the model
            from phase2_subsystems.model_router.model_selector import ModelSelector
            selector = ModelSelector()
            for model in selector.get_available_models():
                if model.name == self._category_model_map[category]:
                    return model
        
        # Find cheapest model that was successful for this category
        category_history = [
            req for req in self._request_history
            if req['category'] == category
        ]
        
        if not category_history:
            return None
        
        # Calculate average cost per request by model
        model_costs: Dict[str, Dict[str, float]] = defaultdict(lambda: {'total': 0.0, 'count': 0})
        for req in category_history:
            model_costs[req['model']]['total'] += req['cost']
            model_costs[req['model']]['count'] += 1
        
        # Find model with lowest average cost
        best_model = None
        best_avg_cost = float('inf')
        
        for model_name, data in model_costs.items():
            avg_cost = data['total'] / data['count']
            if avg_cost < best_avg_cost:
                best_avg_cost = avg_cost
                best_model = model_name
        
        if best_model:
            # Cache the optimized choice
            self._category_model_map[category] = best_model
            
            # Find and return the model
            from phase2_subsystems.model_router.model_selector import ModelSelector
            selector = ModelSelector()
            for model in selector.get_available_models():
                if model.name == best_model:
                    return model
        
        return None
    
    def reset_metrics(self):
        """Reset usage metrics."""
        self._metrics = UsageMetrics()
        self._request_history.clear()
        self._category_model_map.clear()
    
    def get_cost_report(self) -> str:
        """Generate cost report."""
        lines = [
            "=" * 60,
            "Cost Optimization Report",
            "=" * 60,
            f"Total Requests: {self._metrics.total_requests}",
            f"Total Tokens: {self._metrics.total_tokens}",
            f"Total Cost: ${self._metrics.total_cost:.4f}",
            f"Average Latency: {self._metrics.average_latency_ms:.2f}ms",
            "",
            "Usage by Model:",
        ]
        
        for model_name, count in sorted(self._metrics.model_usage.items()):
            lines.append(f"  {model_name}: {count} requests")
        
        lines.append("")
        lines.append("Usage by Category:")
        
        for category, count in sorted(self._metrics.category_usage.items()):
            lines.append(f"  {category}: {count} requests")
        
        if self._category_model_map:
            lines.append("")
            lines.append("Optimized Model Choices:")
            for category, model_name in self._category_model_map.items():
                lines.append(f"  {category}: {model_name}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)

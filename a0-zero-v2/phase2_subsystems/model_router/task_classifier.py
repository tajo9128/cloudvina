"""
Task Classifier - Classifies tasks for optimal model selection.
Uses rule-based and heuristic methods to categorize tasks.
"""
import re
from typing import Optional, Dict, Any, List
from datetime import datetime

import sys
sys.path.insert(0, '/a0/lib')

from phase2_subsystems.model_router.interfaces import (
    ITaskClassifier, TaskCategory, TaskClassification
)


class TaskClassifierConfig:
    """Configuration for TaskClassifier."""
    def __init__(
        self,
        default_confidence: float = 0.85,
        min_confidence: float = 0.7,
        token_estimate_factor: float = 1.2
    ):
        self.default_confidence = default_confidence
        self.min_confidence = min_confidence
        self.token_estimate_factor = token_estimate_factor


class TaskClassifier(ITaskClassifier):
    """Rule-based task classifier for model routing."""
    
    # Task category patterns
    PATTERNS = {
        TaskCategory.PLANNING: [
            r'plan', r'design', r'architect', r'strategy', r'outline',
            r'requirement', r'spec', r'roadmap', r'blueprint'
        ],
        TaskCategory.CODE_GEN: [
            r'generate', r'create', r'implement', r'write code',
            r'build', r'develop', r'program', r'script'
        ],
        TaskCategory.REFACTOR: [
            r'refactor', r'improve', r'optimize', r'clean up',
            r'restructure', r'reorganize', r'conform'
        ],
        TaskCategory.DEBUG: [
            r'debug', r'fix', r'error', r'bug', r'troubleshoot',
            r'issue', r'problem', r'fail', r'exception'
        ],
        TaskCategory.TEST: [
            r'test', r'verify', r'validate', r'check', r'assert',
            r'unit test', r'integration test', r'mock'
        ],
        TaskCategory.DOCS: [
            r'document', r'readme', r'comment', r'explain',
            r'tutorial', r'guide', r'wiki'
        ],
        TaskCategory.QUERY: [
            r'what is', r'how to', r'why', r'explain briefly',
            r'summarize', r'show me', r'list'
        ]
    }
    
    def __init__(self, config: Optional[TaskClassifierConfig] = None):
        self.config = config or TaskClassifierConfig()
        
        # Compile patterns
        self._compiled_patterns = {
            category: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for category, patterns in self.PATTERNS.items()
        }
    
    def classify(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> TaskClassification:
        """Classify task into category."""
        task_lower = task_description.lower()
        context = context or {}
        
        # Score each category
        scores = {}
        for category, patterns in self._compiled_patterns.items():
            score = sum(1 for pattern in patterns if pattern.search(task_lower))
            scores[category] = score
        
        # Find best category
        best_category = max(scores.items(), key=lambda x: x[1])
        
        # Calculate confidence based on score difference
        total_score = sum(scores.values())
        if total_score > 0:
            confidence = self.config.min_confidence + (best_category[1] / total_score) * 0.15
        else:
            confidence = self.config.default_confidence
            best_category = (TaskCategory.QUERY, 0)
        
        # Get alternatives
        alternatives = [
            cat for cat, score in scores.items()
            if cat != best_category[0] and score > 0
        ]
        
        # Determine requirements
        requires_vision = self._check_vision_requirement(task_description, context)
        requires_code = self._check_code_requirement(task_description, context)
        
        # Estimate tokens
        estimated_tokens = self.estimate_tokens(task_description, context)
        
        return TaskClassification(
            category=best_category[0],
            confidence=min(confidence, 1.0),
            reason=f"Matched {best_category[1]} pattern(s) for {best_category[0].value}",
            alternative_categories=alternatives[:2],
            estimated_tokens=estimated_tokens,
            requires_vision=requires_vision,
            requires_code=requires_code
        )
    
    def estimate_tokens(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> int:
        """Estimate token usage for task."""
        context = context or {}
        
        # Base estimate from description
        base_tokens = len(task_description.split()) * 1.5
        
        # Add context size
        context_size = sum(
            len(str(v).split()) * 1.5
            for v in context.values()
            if isinstance(v, (str, int, float, bool))
        )
        
        # Add context from files/attachments
        if 'files' in context:
            file_size = sum(
                len(str(v).split()) * 1.5
                for v in context['files']
            )
            context_size += file_size
        
        # Apply factor and round
        total = int((base_tokens + context_size) * self.config.token_estimate_factor)
        return max(total, 100)
    
    def _check_vision_requirement(self, task: str, context: Dict[str, Any]) -> bool:
        """Check if task requires vision capabilities."""
        vision_keywords = ['image', 'picture', 'screenshot', 'diagram', 'chart', 'visual']
        task_lower = task.lower()
        
        if any(kw in task_lower for kw in vision_keywords):
            return True
        
        # Check context for images
        if 'attachments' in context:
            for att in context['attachments']:
                if isinstance(att, str) and any(
                    ext in att.lower() for ext in ['.png', '.jpg', '.jpeg', '.gif', '.pdf']
                ):
                    return True
        
        return False
    
    def _check_code_requirement(self, task: str, context: Dict[str, Any]) -> bool:
        """Check if task requires code generation/analysis."""
        code_keywords = ['code', 'function', 'class', 'script', 'program', 'api',
                       'python', 'javascript', 'java', 'rust', 'go']
        task_lower = task.lower()
        
        return any(kw in task_lower for kw in code_keywords)

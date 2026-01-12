"""Smart GLM Usage - Avoids LLM for deterministic work."""
from typing import List
from enum import Enum

class TaskCategory(Enum):
    DETERMINISTIC = "deterministic"
    REASONING = "reasoning"
    CREATIVE = "creative"

class DeterministicTaskDetector:
    def __init__(self):
        self._deterministic_patterns = ["list files", "read file", "write file", "git status", "check status"]
        self._reasoning_patterns = ["plan", "design", "analyze", "refactor", "explain"]
        self._creative_patterns = ["create", "generate", "write", "implement", "build"]
    
    def classify(self, task: str) -> TaskCategory:
        task_lower = task.lower()
        for pattern in self._deterministic_patterns:
            if pattern in task_lower:
                return TaskCategory.DETERMINISTIC
        for pattern in self._reasoning_patterns:
            if pattern in task_lower:
                return TaskCategory.REASONING
        for pattern in self._creative_patterns:
            if pattern in task_lower:
                return TaskCategory.CREATIVE
        return TaskCategory.REASONING
    
    def needs_llm(self, task: str) -> bool:
        return self.classify(task) != TaskCategory.DETERMINISTIC

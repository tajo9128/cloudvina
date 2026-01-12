"""
Phase 1 Speed Optimization Layer.

Implements:
1. Aggressive Caching
2. Async Operations
3. Deferred Explainability
4. State Streaming

Expected gains: 40-50% latency reduction, 1.8× perceived speed
"""

from .cache_manager import CacheManager
from .async_operations import AsyncRepoInitializer
from .state_streaming import StateStreamer
from .async_logger import AsyncLogger

__all__ = [
    'CacheManager',
    'AsyncRepoInitializer', 
    'StateStreamer',
    'AsyncLogger'
]

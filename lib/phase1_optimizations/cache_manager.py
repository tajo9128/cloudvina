"""
Cache Manager for Aggressive Caching Optimization.

Implements content-hash based caching with TTL for:
- Repo file index
- Dependency graph
- Repo summary
- Model planning outputs
- Tool schemas

Expected gain: 20-80% faster on repeat tasks
"""

import hashlib
import json
import time
import threading
from typing import Any, Optional, Callable, Dict
from dataclasses import dataclass
from functools import wraps
import pickle


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    timestamp: float
    ttl: Optional[float] = None  # Seconds until expiry
    hits: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl


class CacheManager:
    """
    Content-hash based cache manager with TTL and statistics.
    
    Provides:
    - Content-hash based keys
    - Time-to-live (TTL) expiration
    - Cache statistics
    - Thread-safe operations
    - Pattern-based invalidation
    """
    
    def __init__(self, default_ttl: float = 300.0):  # 5 minutes default
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._default_ttl = default_ttl
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'invalidations': 0
        }
    
    def generate_key(self, content: str, namespace: str = '') -> str:
        """
        Generate content-hash cache key.
        
        Args:
            content: Content to hash
            namespace: Optional namespace for key
        
        Returns:
            SHA256 hash of content
        """
        key_string = f"{namespace}:{content}" if namespace else content
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value if exists and not expired, None otherwise
        """
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._stats['misses'] += 1
                return None
            
            if entry.is_expired():
                # Expired, remove it
                del self._cache[key]
                self._stats['misses'] += 1
                self._stats['evictions'] += 1
                return None
            
            # Cache hit
            entry.hits += 1
            self._stats['hits'] += 1
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        with self._lock:
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                ttl=ttl if ttl is not None else self._default_ttl
            )
            self._cache[key] = entry
    
    def get_or_compute(self, key: str, compute_fn: Callable, ttl: Optional[float] = None) -> Any:
        """
        Get from cache or compute if not present.
        
        Args:
            key: Cache key
            compute_fn: Function to compute value if not cached
            ttl: Time-to-live for cached value
        
        Returns:
            Cached or computed value
        """
        # Try cache first
        value = self.get(key)
        if value is not None:
            return value
        
        # Compute value
        value = compute_fn()
        
        # Cache it
        self.set(key, value, ttl)
        
        return value
    
    def invalidate(self, pattern: str = None) -> None:
        """
        Invalidate cache entries by pattern.
        
        Args:
            pattern: Pattern to match (None = clear all)
        """
        with self._lock:
            if pattern is None:
                # Clear all
                count = len(self._cache)
                self._cache.clear()
                self._stats['invalidations'] += count
            else:
                # Pattern-based invalidation
                keys_to_delete = [k for k in self._cache.keys() if pattern in k]
                for k in keys_to_delete:
                    del self._cache[k]
                    self._stats['invalidations'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = (self._stats['hits'] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                **self._stats,
                'hit_rate': hit_rate,
                'cache_size': len(self._cache)
            }
    
    def reset_stats(self) -> None:
        """Reset cache statistics."""
        with self._lock:
            self._stats = {
                'hits': 0,
                'misses': 0,
                'evictions': 0,
                'invalidations': 0
            }


def cached(ttl: float = 300.0, key_prefix: str = ''):
    """
    Decorator to cache function results.
    
    Args:
        ttl: Time-to-live in seconds
        key_prefix: Prefix for cache key
    """
    _manager = CacheManager(default_ttl=ttl)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from function arguments
            key_parts = [key_prefix, func.__name__, str(args), str(kwargs)]
            key = _manager.generate_key(":".join(key_parts))
            
            # Get or compute
            def compute():
                return func(*args, **kwargs)
            
            return _manager.get_or_compute(key, compute, ttl=ttl)
        
        wrapper._cache_manager = _manager  # Expose manager for stats
        return wrapper
    
    return decorator


# Global cache manager instance
_global_cache = CacheManager()


def get_global_cache() -> CacheManager:
    """Get global cache manager instance."""
    return _global_cache

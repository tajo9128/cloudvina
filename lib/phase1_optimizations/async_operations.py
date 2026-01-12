"""
Async Operations for Parallel Initialization.

Implements parallel execution of:
- File indexing
- Dependency graph building
- State loading

Expected gain: 300-800ms saved on cold start
"""

import asyncio
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, '/a0/lib')


@dataclass
class ParallelInitResult:
    """Result of parallel initialization."""
    success: bool
    elapsed_time: float
    results: Dict[str, Any]
    errors: Dict[str, Exception] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = {}


class AsyncRepoInitializer:
    """
    Async initializer for repo awareness components.
    
    Runs file indexing, dependency graph building, and state loading
    in parallel to minimize cold start latency.
    """
    
    def __init__(self, project_path: str):
        self.project_path = project_path
        self._indexer = None
        self._graph_builder = None
        self._state_manager = None
    
    async def initialize_all(self) -> ParallelInitResult:
        """
        Initialize all repo awareness components in parallel.
        
        Returns:
            ParallelInitResult with results and timing
        """
        start_time = time.perf_counter()
        results = {}
        errors = {}
        success = True
        
        try:
            # Create tasks for parallel execution
            tasks = [
                self._index_repo_async(),
                self._build_graph_async(),
                self._load_state_async()
            ]
            
            # Run all in parallel and wait for completion
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            component_names = ['indexer', 'graph', 'state']
            for i, result in enumerate(task_results):
                name = component_names[i]
                
                if isinstance(result, Exception):
                    errors[name] = result
                    success = False
                else:
                    results[name] = result
        
        except Exception as e:
            errors['general'] = e
            success = False
        
        elapsed_time = time.perf_counter() - start_time
        
        return ParallelInitResult(
            success=success,
            elapsed_time=elapsed_time,
            results=results,
            errors=errors
        )
    
    async def _index_repo_async(self):
        """Async file indexing."""
        # Import here to avoid circular dependency
        from repo_awareness.file_indexer import FileIndexer
        
        def index_sync():
            indexer = FileIndexer(self.project_path)
            return indexer.index()
        
        # Run blocking operation in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, index_sync)
    
    async def _build_graph_async(self):
        """Async dependency graph building."""
        from repo_awareness.dependency_graph import DependencyGraphBuilder
        
        def build_sync():
            builder = DependencyGraphBuilder(self.project_path)
            return builder.build()
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, build_sync)
    
    async def _load_state_async(self):
        """Async state loading."""
        from project_state.state_manager import ProjectStateManager
        
        def load_sync():
            manager = ProjectStateManager(self.project_path)
            return manager.load_state()
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, load_sync)
    
    @property
    def indexer(self):
        """Get file indexer instance."""
        return self._indexer
    
    @property
    def graph_builder(self):
        """Get graph builder instance."""
        return self._graph_builder
    
    @property
    def state_manager(self):
        """Get state manager instance."""
        return self._state_manager


async def initialize_repo_async(project_path: str) -> ParallelInitResult:
    """
    Convenience function to initialize repo asynchronously.
    
    Args:
        project_path: Path to project
    
    Returns:
        ParallelInitResult with timing and results
    """
    initializer = AsyncRepoInitializer(project_path)
    return await initializer.initialize_all()


# Benchmark function
def benchmark_parallel_vs_sequential(project_path: str) -> Dict[str, float]:
    """
    Benchmark parallel vs sequential initialization.
    
    Returns:
        Dict with timing for parallel and sequential execution
    """
    import time
    
    results = {}
    
    # Benchmark sequential
    print("Benchmarking sequential initialization...")
    start = time.perf_counter()
    
    try:
        from repo_awareness.file_indexer import FileIndexer
        from repo_awareness.dependency_graph import DependencyGraphBuilder
        from project_state.state_manager import ProjectStateManager
        
        # Sequential execution
        indexer = FileIndexer(project_path)
        indexer.index()
        
        builder = DependencyGraphBuilder(project_path)
        builder.build()
        
        manager = ProjectStateManager(project_path)
        manager.load_state()
        
    except Exception as e:
        print(f"Sequential error: {e}")
    
    sequential_time = time.perf_counter() - start
    results['sequential_time'] = sequential_time
    print(f"Sequential time: {sequential_time:.3f}s")
    
    # Benchmark parallel
    print("Benchmarking parallel initialization...")
    start = time.perf_counter()
    
    try:
        loop = asyncio.get_event_loop()
        parallel_result = loop.run_until_complete(initialize_repo_async(project_path))
        parallel_time = parallel_result.elapsed_time
    except Exception as e:
        print(f"Parallel error: {e}")
        parallel_time = 0
    
    results['parallel_time'] = parallel_time
    print(f"Parallel time: {parallel_time:.3f}s")
    
    # Calculate speedup
    if parallel_time > 0 and sequential_time > 0:
        speedup = sequential_time / parallel_time
        time_saved = sequential_time - parallel_time
        results['speedup'] = speedup
        results['time_saved'] = time_saved
        print(f"Speedup: {speedup:.2f}x")
        print(f"Time saved: {time_saved:.3f}s")
    
    return results


if __name__ == "__main__":
    # Test with current project
    project_path = "/a0/usr/projects/a0-cloudvina"
    
    if os.path.exists(project_path):
        print("Testing async operations...")
        print("=" * 50)
        results = benchmark_parallel_vs_sequential(project_path)
        print("=" * 50)
        print(f"\nResults: {results}")
    else:
        print(f"Project path not found: {project_path}")

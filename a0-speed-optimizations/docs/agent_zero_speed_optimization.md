# Agent Zero Speed Optimization Map
## Implementing Google Antigravity-Style Responsiveness

**Version**: v2.1 (Speed Optimization Layer)
**Date**: 2026-01-12
**Status**: 📋 Implementation Plan

---

# Overview

This document maps **8 speed optimizations** from the Antigravity guide onto **existing Agent Zero v2 modules** (Phase 1 + Phase 2).

**Expected Speed Gain**: **2-3× faster perceived speed**, 40-80% faster actual latency.

---

# Optimization Mapping Table

| # | Optimization | Primary Module | Secondary Module | Expected Gain |
|---|--------------|----------------|------------------|---------------|
| 1 | Reduce LLM Calls | ModelRouter (Phase 2B) | RepoAwareness (P1) | 40-60% |
| 2 | Aggressive Caching | ProjectStateManager (P1) | FileIndexer (P1) | 20-80% |
| 3 | Parallelization | RepoAwareness (P1) | All modules | 300-800 ms |
| 4 | Defer Explainability | ExecutionTracer (P1) | DecisionLogger (P1) | Huge gain |
| 5 | Stream States | DecisionLogger (P1) | All modules | 2× perceived |
| 6 | Smart GLM Usage | ModelRouter (P2B) | All tools | Cost + Speed |
| 7 | Fast vs Deep Path | ModelRouter (P2B) | GitGovernor (P2A) | 30-50% |
| 8 | Connection Opt | All modules | - | 50-200 ms |

---

# Detailed Implementation Maps

## 1️⃣ Reduce LLM Calls (Biggest Win: 40-60%)

### Current Slow Pattern
```
User → Plan (LLM)
     → Re-check repo (LLM)
     → Decide tool (LLM)
     → Execute
     → Explain (LLM)
```
**4-6 LLM calls per task**

### Optimized Pattern (Single-Call Planning Envelope)
```
User → One structured reasoning call (LLM)
     → Local execution loop
     → Optional refinement (LLM only if needed)
```
**1-2 LLM calls per task**

---

### Module Integration

#### **Primary: ModelRouter (Phase 2B)**

**File**: `/a0/lib/phase2_subsystems/model_router/model_router.py`

**Changes Required**:
```python
class ModelRouter:
    
    def route_with_envelope(self, task: str, context: dict = None) -> RoutingDecision:
        """
        Single-call planning envelope: Get plan, tool sequence, and risk in one LLM call.
        """
        # Prepare enriched context
        envelope = {
            "task": task,
            "repo_summary": self._get_cached_repo_summary(),
            "constraints": self._get_safety_constraints(),
            "required_outputs": ["plan", "tool_sequence", "risk_assessment"],
            "model_preference": "fast"  # Ask for fast reasoning
        }
        
        # Single GLM-4.7 call produces all outputs
        response = self._call_llm_structured(envelope)
        
        # Parse multi-part response
        plan = response.get("plan")
        tool_sequence = response.get("tool_sequence")
        risk = response.get("risk_assessment")
        
        return RoutingDecision(
            task_id=task,
            classification=self._classify_task(task),
            selection=self._select_optimal_model(task),
            plan=plan,
            tool_sequence=tool_sequence,
            risk_assessment=risk
        )
```

#### **Secondary: RepoAwareness (Phase 1)**

**File**: `/a0/lib/repo_awareness/file_indexer.py`

**Changes Required**:
```python
class FileIndexer:
    
    def get_context_for_envelope(self, task: str) -> dict:
        """
        Provide pre-computed context to avoid redundant LLM calls.
        """
        # Return from cache, don't call LLM
        return {
            "repo_summary": self.cached_summary,
            "relevant_files": self.query(task),
            "dependency_context": self.get_dependencies_for_files(task)
        }
```

---

### Impact
- **Latency Reduction**: 40-60%
- **Token Savings**: 50-70%
- **User Perception**: Much faster

---

## 2️⃣ Aggressive Caching (20-80% on Repeat Tasks)

### Cache Strategy

| Cache Item | TTL | Module | Trigger Invalidation |
|-------------|-----|--------|---------------------|
| Repo file index | Until file change | FileIndexer | File watch event |
| Dependency graph | Until index change | DependencyGraph | Index rebuild |
| Repo summary | Long (1 hour) | ProjectStateManager | Manual refresh |
| Model planning outputs | Short (5 min) | ModelRouter | New context hash |
| Tool schemas | Forever | All modules | Never |

---

### Module Integration

#### **Primary: ProjectStateManager (Phase 1)**

**File**: `/a0/lib/project_state/state_manager.py`

**Changes Required**:
```python
class ProjectStateManager:
    
    def __init__(self):
        self.cache = {}  # content_hash -> value
        self.cache_metadata = {}  # content_hash -> {timestamp, ttl}
    
    def get_or_compute(self, key: str, compute_fn, ttl_seconds: int = None):
        """
        Content-hash based caching with TTL.
        """
        import hashlib
        import time
        
        # Check cache
        if key in self.cache:
            metadata = self.cache_metadata[key]
            if ttl_seconds is None or time.time() - metadata['timestamp'] < ttl_seconds:
                return self.cache[key]
        
        # Compute and cache
        value = compute_fn()
        self.cache[key] = value
        self.cache_metadata[key] = {
            'timestamp': time.time(),
            'ttl': ttl_seconds
        }
        return value
    
    def invalidate(self, pattern: str = None):
        """
        Invalidate cache by pattern.
        """
        if pattern:
            keys_to_delete = [k for k in self.cache if pattern in k]
            for k in keys_to_delete:
                del self.cache[k]
                del self.cache_metadata[k]
        else:
            self.cache.clear()
            self.cache_metadata.clear()
```

#### **Secondary: FileIndexer (Phase 1)**

**File**: `/a0/lib/repo_awareness/file_indexer.py`

**Changes Required**:
```python
class FileIndexer:
    
    def __init__(self, project_path: str):
        self.project_path = project_path
        self.state_manager = ProjectStateManager()
        self._file_watch_handle = None
        
        # Start file watcher
        self._start_file_watcher()
    
    def _start_file_watcher(self):
        """
        Watch for file changes to invalidate cache.
        """
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
        
        class FileChangeHandler(FileSystemEventHandler):
            def __init__(self, indexer):
                self.indexer = indexer
            
            def on_modified(self, event):
                if not event.is_directory:
                    # Invalidate related cache entries
                    self.indexer.state_manager.invalidate(event.src_path)
        
        observer = Observer()
        observer.schedule(FileChangeHandler(self), self.project_path, recursive=True)
        observer.start()
        self._file_watch_handle = observer
```

---

### Impact
- **First Task**: Normal speed
- **Repeat Tasks**: 20-80% faster (instant for cached)
- **Memory**: Minimal (text-based caches)

---

## 3️⃣ Parallelize Non-LLM Operations (300-800 ms saved)

### Current (Slow)
```
Scan repo → Build graph → Load state → Ask LLM → Execute

Total: 1500-2500 ms
```

### Optimized (Antigravity-style)
```
Scan repo ─┐
Build graph ├─→ Ready state (parallel)
Load state ─┘

Ask LLM (only after)

Total: 800-1200 ms (saved 700-1300 ms)
```

---

### Module Integration

#### **Primary: RepoAwareness (Phase 1)**

**File**: `/a0/lib/repo_awareness/__init__.py`

**Changes Required**:
```python
import asyncio

async def initialize_repo_awareness_parallel(project_path: str):
    """
    Initialize all repo awareness modules in parallel.
    """
    indexer = FileIndexer(project_path)
    graph_builder = DependencyGraphBuilder(project_path)
    state_manager = ProjectStateManager()
    
    # Run all in parallel
    results = await asyncio.gather(
        indexer.index_async(),
        graph_builder.build_async(),
        state_manager.load_async()
    )
    
    index_result, graph_result, state_result = results
    
    return {
        'indexer': indexer,
        'graph_builder': graph_builder,
        'state_manager': state_manager
    }

class FileIndexer:
    
    async def index_async(self):
        """
        Async version of index() for parallel execution.
        """
        # Use asyncio ThreadPoolExecutor for I/O-bound work
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.index)

class DependencyGraphBuilder:
    
    async def build_async(self):
        """
        Async version of build() for parallel execution.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.build)
```

---

### Impact
- **Cold Start**: 300-800 ms faster
- **Perceived Speed**: Immediate "ready" state
- **Resource**: Minimal (parallel threads)

---

## 4️⃣ Defer Explainability (Huge Perceived Speed Gain)

### Current (Slow)
```
Decide → Execute → Explain (LLM call) → Respond

User waits for explanation
```

### Optimized
```
Decide → Execute → Respond immediately
               ↓
         Log explanation asynchronously

User gets instant response
```

---

### Module Integration

#### **Primary: ExecutionTracer (Phase 1)**

**File**: `/a0/lib/explainability/execution_tracer.py`

**Changes Required**:
```python
import asyncio

class ExecutionTracer:
    
    def __init__(self):
        self.trace_queue = asyncio.Queue()
        self._background_task = None
        
        # Start background processor
        self._start_background_processor()
    
    async def log_async(self, event: ExecutionEvent):
        """
        Non-blocking log - adds to queue, returns immediately.
        """
        await self.trace_queue.put(event)
        # Don't wait for write
    
    def _start_background_processor(self):
        """
        Background task writes traces to disk.
        """
        async def _process_queue():
            while True:
                event = await self.trace_queue.get()
                await self._write_event(event)
        
        self._background_task = asyncio.create_task(_process_queue())
    
    async def _write_event(self, event: ExecutionEvent):
        """
        Actually write to disk (runs in background).
        """
        # Write to file/database
        pass
```

#### **Secondary: DecisionLogger (Phase 1)**

**File**: `/a0/lib/explainability/decision_logger.py`

**Changes Required**:
```python
class DecisionLogger:
    
    def log_decision_async(self, decision: RoutingDecision):
        """
        Log decision in background, don't block.
        """
        asyncio.create_task(self._log_decision_background(decision))
    
    async def _log_decision_background(self, decision: RoutingDecision):
        """
        Actually log decision (runs in background).
        """
        # Write to log file
        # Can include LLM call for explanation here (non-blocking)
        pass
```

---

### Impact
- **User Perception**: Instant responses
- **LLM Calls**: Same (just moved to background)
- **Explainability**: Still complete, just not blocking

---

## 5️⃣ Stream Agent State Events (2× Perceived Speed)

### Current
```
User sends task → [silence for 3-5 seconds] → Result
```

### Optimized (Antigravity-style)
```
User sends task → "Planning..." → "Checking repo..." → "Executing..." → Result

User sees progress, feels faster
```

---

### Module Integration

#### **Primary: DecisionLogger (Phase 1)**

**File**: `/a0/lib/explainability/decision_logger.py`

**Changes Required**:
```python
from enum import Enum

class AgentState(Enum):
    THINKING = "thinking"
    PLANNING = "planning"
    CHECKING_REPO = "checking_repo"
    EXECUTING = "executing"
    COMPLETED = "completed"
    ERROR = "error"

class DecisionLogger:
    
    def __init__(self):
        self.state_callback = None  # Set by agent to emit to user
    
    def set_state_callback(self, callback):
        """
        Set callback to emit state events to user.
        """
        self.state_callback = callback
    
    def emit_state(self, state: AgentState, details: str = None):
        """
        Emit state event to user (streaming).
        """
        event = {
            "state": state.value,
            "details": details,
            "timestamp": time.time()
        }
        
        if self.state_callback:
            self.state_callback(event)
```

#### **Secondary: All Modules**

**Usage Example**:
```python
# In agent execution loop:
decision_logger.emit_state(AgentState.PLANNING, "Creating plan...")

# Do planning...

decision_logger.emit_state(AgentState.CHECKING_REPO, "Scanning files...")

# Do repo check...

decision_logger.emit_state(AgentState.EXECUTING, "Running tools...")

# Execute...

decision_logger.emit_state(AgentState.COMPLETED)
```

---

### Impact
- **Perceived Speed**: 2× faster
- **Actual Speed**: Same
- **User Satisfaction**: Much higher

---

## 6️⃣ Smart GLM-4.7 Usage (Cost + Speed)

### GLM-4.7 Strengths
- ✅ Planning
- ✅ Reasoning
- ✅ Code generation

### GLM-4.7 Weaknesses (Don't Use For)
- ❌ File I/O
- ❌ Diffs
- ❌ Formatting
- ❌ Logging
- ❌ Static templates

---

### Module Integration

#### **Primary: ModelRouter (Phase 2B)**

**File**: `/a0/lib/phase2_subsystems/model_router/task_classifier.py`

**Changes Required**:
```python
class TaskClassifier:
    
    def classify(self, task: str) -> TaskClassification:
        """
        Classify task and determine if LLM is needed.
        """
        # Check if task is deterministic
        if self._is_deterministic(task):
            # Don't use LLM, use local logic
            return TaskClassification(
                category=TaskCategory.DETERMINISTIC,
                confidence=1.0,
                reason="Deterministic task, no LLM needed"
            )
        
        # Otherwise, use LLM for classification
        return self._classify_with_llm(task)
    
    def _is_deterministic(self, task: str) -> bool:
        """
        Check if task is deterministic (file I/O, diffs, formatting).
        """
        deterministic_patterns = [
            "list files",
            "read file",
            "check status",
            "format code",
            "show diff"
        ]
        
        return any(p in task.lower() for p in deterministic_patterns)
```

---

### Impact
- **LLM Calls**: 30-50% fewer
- **Token Usage**: 40-60% less
- **Cost**: Significant savings
- **Speed**: Faster for deterministic tasks

---

## 7️⃣ Fast vs Deep Path Execution (30-50% faster for common tasks)

### Fast Path (default)
- Small edits
- Known patterns
- Cached plans
- Minimal reasoning

### Deep Path (only when needed)
- Large refactors
- Cross-module changes
- High risk

---

### Module Integration

#### **Primary: ModelRouter (Phase 2B)**

**File**: `/a0/lib/phase2_subsystems/model_router/model_router.py`

**Changes Required**:
```python
class ModelRouter:
    
    def route(self, task: str, context: dict = None) -> RoutingDecision:
        """
        Route task to fast or deep path.
        """
        # Classify complexity
        complexity = self._classify_complexity(task, context)
        
        if complexity == "low":
            # Fast path
            return self._fast_path(task, context)
        else:
            # Deep path
            return self._deep_path(task, context)
    
    def _classify_complexity(self, task: str, context: dict) -> str:
        """
        Classify task complexity.
        """
        # Check cache first
        cache_key = self._cache_key_for_task(task)
        cached = self.state_manager.get(cache_key)
        if cached:
            return cached
        
        # Simple heuristics (no LLM needed)
        if self._is_small_edit(task):
            return "low"
        elif self._is_large_refactor(task, context):
            return "high"
        else:
            return "medium"
    
    def _fast_path(self, task: str, context: dict) -> RoutingDecision:
        """
        Fast path: Use cached plan, minimal reasoning.
        """
        # Check if plan is cached
        cached_plan = self.state_manager.get(f"plan:{task}")
        
        if cached_plan:
            # Reuse cached plan (instant)
            return RoutingDecision(
                task_id=task,
                classification=TaskClassification(
                    category=TaskCategory.CACHED,
                    confidence=1.0,
                    reason="Reusing cached plan"
                ),
                selection=cached_plan.selection,
                plan=cached_plan.plan,
                tool_sequence=cached_plan.tool_sequence,
                execution_path="fast"
            )
        
        # No cache, but still use fast model
        return self._route_to_fast_model(task)
    
    def _deep_path(self, task: str, context: dict) -> RoutingDecision:
        """
        Deep path: Full reasoning, multiple LLM calls.
        """
        # Use best model, full planning envelope
        return self._route_with_envelope(task, context)
```

#### **Secondary: GitGovernor (Phase 2A)**

**File**: `/a0/lib/phase2_subsystems/git_governor/git_governor.py`

**Changes Required**:
```python
class GitGovernor:
    
    def validate_commit(self, commit: CommitRequest) -> ValidationResult:
        """
        Fast path for low-risk commits.
        """
        # Check risk assessment
        risk = self.assessor.assess(commit)
        
        if risk.level == "low":
            # Fast path: Skip expensive validations
            return ValidationResult(
                can_commit=True,
                reason="Low risk, fast-tracked"
            )
        else:
            # Deep path: Full validation
            return self._full_validation(commit)
```

---

### Impact
- **Common Tasks**: 30-50% faster
- **Rare Complex Tasks**: Same speed (necessary safety)
- **User Experience**: Fast for most work

---

## 8️⃣ Connection-Level Optimizations (50-200 ms saved)

### Optimizations
- ✅ Reuse HTTP sessions
- ✅ Enable keep-alive
- ✅ Set aggressive timeouts
- ✅ Disable retries for interactive calls

---

### Module Integration

#### **All LLM-Calling Modules**

**Changes Required**:
```python
import aiohttp

class LLMClient:
    
    def __init__(self):
        # Reuse session
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(
                keepalive_timeout=30,
                limit=100
            ),
            timeout=aiohttp.ClientTimeout(total=20),  # 20s timeout
        )
    
    async def call_llm(self, prompt: str):
        """
        Call LLM with optimized connection.
        """
        # No retries for interactive calls
        async with self.session.post(
            self.api_url,
            json={"prompt": prompt},
            timeout=20  # Aggressive timeout
        ) as response:
            return await response.json()
```

---

### Impact
- **Per-Call Latency**: 50-200 ms saved
- **Cumulative**: Significant over many calls
- **Resource**: Minimal

---

# Implementation Priority

## Phase 1 (Week 1): Critical Path
1. ✅ **Reduce LLM Calls** (40-60% gain) - Highest ROI
2. ✅ **Aggressive Caching** (20-80% on repeat)
3. ✅ **Defer Explainability** (Huge perception gain)

## Phase 2 (Week 2): Speed Layer
4. ✅ **Stream States** (2× perceived speed)
5. ✅ **Parallelization** (300-800 ms saved)
6. ✅ **Fast vs Deep Path** (30-50% for common tasks)

## Phase 3 (Week 3): Optimization
7. ✅ **Smart GLM Usage** (Cost + Speed)
8. ✅ **Connection Optimization** (50-200 ms)

---

# Expected Cumulative Gains

| Phase | Latency Reduction | Perceived Speed |
|-------|-------------------|-----------------|
| Current | 0% | 1× |
| Phase 1 | 40-50% | 1.8× |
| Phase 1+2 | 60-70% | 2.5× |
| Phase 1+2+3 | 70-80% | 3× |

---

# Integration Points

### With Phase 1 Modules
- **FileIndexer**: Add caching, async operations
- **DependencyGraph**: Add async building
- **ProjectStateManager**: Add cache manager
- **DecisionLogger**: Add state streaming
- **ExecutionTracer**: Add async logging

### With Phase 2 Modules
- **ModelRouter**: Add fast/deep path, smart LLM usage
- **GitGovernor**: Add fast path for low-risk commits
- **TaskClassifier**: Add deterministic detection
- **CostOptimizer**: Add cache hit tracking

---

# Testing Strategy

## Performance Benchmarks

```python
# Test 1: Single LLM Call vs Multi-Call
test_single_call_envelope()  # Should be 40-60% faster

# Test 2: Cache Hit vs Miss
test_cache_hit()  # Should be instant (near 0ms)

# Test 3: Parallel vs Sequential
test_parallel_init()  # Should save 300-800ms

# Test 4: Fast Path vs Deep Path
test_fast_path()  # Should be 30-50% faster

# Test 5: Deferred Explainability
test_async_logging()  # Should not block response
```

---

# Success Criteria

- [ ] LLM calls per task reduced by 50% (4→2 calls)
- [ ] Cache hit rate > 60% on repeat tasks
- [ ] Parallel init saves > 500ms on cold start
- [ ] Explainability is non-blocking (async)
- [ ] State events stream to user (no silence)
- [ ] Fast path used for > 70% of tasks
- [ ] Connection latency < 100ms
- [ ] Overall perceived speed > 2.5×

---

**Implementation Status**: 📋 Ready for development

**Next Action**: Implement Phase 1 optimizations (Critical Path)

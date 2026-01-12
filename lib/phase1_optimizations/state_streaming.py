"""
State Streaming for Real-Time Status Updates.

Emits agent state events to user for 2× perceived speed.

Example events:
- "Planning..."
- "Checking repo..."
- "Executing..."

Expected gain: 2× perceived speed (user sees progress)
"""

import asyncio
import json
import time
from enum import Enum
from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass, asdict
from collections import deque


class AgentState(Enum):
    """Agent execution states."""
    IDLE = "idle"
    THINKING = "thinking"
    PLANNING = "planning"
    CHECKING_REPO = "checking_repo"
    BUILDING_DEPENDENCIES = "building_dependencies"
    EXECUTING = "executing"
    VALIDATING = "validating"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class StateEvent:
    """Agent state event."""
    state: str
    details: Optional[str] = None
    progress: Optional[float] = None  # 0.0 to 1.0
    timestamp: float = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class StateStreamer:
    """
    Streams agent state events to user in real-time.
    
    Provides:
    - State change notifications
    - Progress tracking
    - Event history
    - Callback registration
    - Async event queue
    """
    
    def __init__(self, max_history: int = 100):
        self._current_state = AgentState.IDLE
        self._callbacks: List[Callable[[StateEvent], None]] = []
        self._history: deque = deque(maxlen=max_history)
        self._event_queue: asyncio.Queue = None
        self._queue_task: asyncio.Task = None
        self._lock = asyncio.Lock()
    
    def start_queue(self) -> None:
        """Start async event queue processing."""
        self._event_queue = asyncio.Queue()
        self._queue_task = asyncio.create_task(self._process_queue())
    
    async def _process_queue(self) -> None:
        """Process events from queue (runs in background)."""
        while True:
            event = await self._event_queue.get()
            await self._emit_async(event)
    
    async def stop_queue(self) -> None:
        """Stop async event queue processing."""
        if self._queue_task:
            self._queue_task.cancel()
            try:
                await self._queue_task
            except asyncio.CancelledError:
                pass
    
    def register_callback(self, callback: Callable[[StateEvent], None]) -> None:
        """
        Register callback for state events.
        
        Args:
            callback: Function to call with state events
        """
        if callback not in self._callbacks:
            self._callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable[[StateEvent], None]) -> None:
        """
        Unregister callback.
        
        Args:
            callback: Callback to remove
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def emit(self, state: AgentState, details: str = None, 
             progress: float = None, **metadata) -> None:
        """
        Emit state event to all callbacks (synchronous).
        
        Args:
            state: Agent state
            details: Optional details message
            progress: Optional progress (0.0 to 1.0)
            **metadata: Additional metadata
        """
        event = StateEvent(
            state=state.value,
            details=details,
            progress=progress,
            metadata=metadata
        )
        
        self._history.append(event)
        self._current_state = state
        
        # Notify all callbacks
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                # Don't let callback errors break streaming
                print(f"Callback error: {e}")
    
    async def emit_async(self, event: StateEvent) -> None:
        """
        Emit state event asynchronously.
        
        Args:
            event: State event to emit
        """
        self._history.append(event)
        
        # Notify all callbacks (async)
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                print(f"Callback error: {e}")
    
    def queue_event(self, state: AgentState, details: str = None,
                    progress: float = None, **metadata) -> None:
        """
        Queue event for async processing (non-blocking).
        
        Args:
            state: Agent state
            details: Optional details message
            progress: Optional progress (0.0 to 1.0)
            **metadata: Additional metadata
        """
        if self._event_queue is None:
            self.start_queue()
        
        event = StateEvent(
            state=state.value,
            details=details,
            progress=progress,
            metadata=metadata
        )
        
        self._event_queue.put_nowait(event)
    
    @property
    def current_state(self) -> AgentState:
        """Get current agent state."""
        return self._current_state
    
    @property
    def history(self) -> List[StateEvent]:
        """Get event history."""
        return list(self._history)
    
    def get_history_json(self) -> str:
        """Get event history as JSON."""
        events = [e.to_dict() for e in self._history]
        return json.dumps(events, indent=2)


# Global state streamer
_global_streamer = StateStreamer()


def get_global_streamer() -> StateStreamer:
    """Get global state streamer instance."""
    return _global_streamer


# Context manager for task states
from contextlib import contextmanager

@contextmanager
def task_state(state: AgentState, details: str = None, 
               progress: float = None, **metadata):
    """
    Context manager for task state.
    
    Usage:
        with task_state(AgentState.PLANNING, "Creating plan..."):
            # Do planning
            pass
    """
    streamer = get_global_streamer()
    streamer.emit(state, details, progress, **metadata)
    try:
        yield
    finally:
        # Return to idle or error state
        if streamer.current_state == state:
            streamer.emit(AgentState.IDLE)


if __name__ == "__main__":
    # Test state streaming
    print("Testing StateStreamer...")
    
    streamer = StateStreamer()
    
    # Add callback to print events
    def print_event(event: StateEvent):
        print(f"[{event.state}] {event.details} (progress: {event.progress})")
    
    streamer.register_callback(print_event)
    
    # Emit some events
    streamer.emit(AgentState.THINKING, "Starting task...")
    streamer.emit(AgentState.PLANNING, "Creating plan...", progress=0.2)
    streamer.emit(AgentState.CHECKING_REPO, "Scanning files...", progress=0.4)
    streamer.emit(AgentState.EXECUTING, "Running tools...", progress=0.7)
    streamer.emit(AgentState.COMPLETED, "Task completed!", progress=1.0)
    
    # Show history
    print("\nEvent History:")
    print(streamer.get_history_json())

"""
Async Logger for Deferred Explainability.

Logs decisions and executions asynchronously so they don't
block the user response.

Expected gain: Huge perceived speed improvement
"""

import asyncio
import json
import os
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading


class LogEntryType(Enum):
    """Types of log entries."""
    DECISION = "decision"
    EXECUTION = "execution"
    ERROR = "error"
    EXPLANATION = "explanation"
    AUDIT = "audit"


@dataclass
class LogEntry:
    """Async log entry."""
    entry_type: str
    content: Dict[str, Any]
    timestamp: float = None
    session_id: str = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.session_id is None:
            self.session_id = os.getenv('SESSION_ID', 'default')
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'entry_type': self.entry_type,
            'content': self.content,
            'timestamp': self.timestamp,
            'session_id': self.session_id,
            'metadata': self.metadata
        }


class AsyncLogger:
    """
    Async logger for non-blocking explainability.
    
    Features:
    - Non-blocking log writes (queued)
    - Background writer task
    - Multiple log levels
    - Session-based logs
    - JSON file output
    """
    
    def __init__(self, log_dir: str = "/a0/tmp/chats/logs",
                 max_queue_size: int = 1000):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._writer_task: asyncio.Task = None
        self._running = False
        self._lock = threading.Lock()
        
        # Start background writer
        self._start()
    
    def _start(self) -> None:
        """Start background writer task."""
        if self._running:
            return
        
        self._running = True
        loop = asyncio.get_event_loop()
        self._writer_task = loop.create_task(self._write_loop())
    
    async def _write_loop(self) -> None:
        """Background loop that writes log entries to disk."""
        while self._running:
            try:
                # Get entry from queue (with timeout to check running flag)
                entry = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                await self._write_entry(entry)
            except asyncio.TimeoutError:
                # Timeout is expected, just check running flag
                continue
            except Exception as e:
                print(f"Async logger error: {e}")
    
    async def _write_entry(self, entry: LogEntry) -> None:
        """
        Write log entry to disk.
        
        Args:
            entry: Log entry to write
        """
        # Create log file path based on session and type
        timestamp_str = time.strftime("%Y%m%d")
        log_file = self.log_dir / f"{entry.session_id}_{timestamp_str}_{entry.entry_type}.log"
        
        # Append entry to file
        with open(log_file, 'a') as f:
            f.write(json.dumps(entry.to_dict()) + "\n")
    
    async def stop(self) -> None:
        """Stop background writer."""
        self._running = False
        
        if self._writer_task:
            self._writer_task.cancel()
            try:
                await self._writer_task
            except asyncio.CancelledError:
                pass
    
    def log(self, entry_type: LogEntryType, content: Dict[str, Any],
           session_id: str = None, **metadata) -> None:
        """
        Log entry asynchronously (non-blocking).
        
        Args:
            entry_type: Type of log entry
            content: Log content
            session_id: Session ID
            **metadata: Additional metadata
        """
        entry = LogEntry(
            entry_type=entry_type.value,
            content=content,
            session_id=session_id,
            metadata=metadata
        )
        
        # Add to queue (non-blocking)
        try:
            self._queue.put_nowait(entry)
        except asyncio.QueueFull:
            print("Async logger queue full, dropping entry")
    
    async def log_async(self, entry_type: LogEntryType, content: Dict[str, Any],
                       session_id: str = None, **metadata) -> None:
        """
        Log entry asynchronously (awaitable version).
        
        Args:
            entry_type: Type of log entry
            content: Log content
            session_id: Session ID
            **metadata: Additional metadata
        """
        entry = LogEntry(
            entry_type=entry_type.value,
            content=content,
            session_id=session_id,
            metadata=metadata
        )
        
        await self._queue.put(entry)


# Global async logger
_global_logger: Optional[AsyncLogger] = None

def get_global_logger() -> AsyncLogger:
    """Get global async logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = AsyncLogger()
    return _global_logger


if __name__ == "__main__":
    # Test async logger
    print("Testing AsyncLogger...")
    
    async def test_logger():
        logger = AsyncLogger()
        
        # Log some entries
        logger.log(LogEntryType.DECISION, {
            'task': 'test_task',
            'decision': 'use_tool_x'
        })
        
        logger.log(LogEntryType.EXECUTION, {
            'tool': 'code_execution_tool',
            'result': 'success'
        })
        
        logger.log(LogEntryType.EXPLANATION, {
            'reasoning': 'Chose tool X because...'
        })
        
        # Wait for queue to drain
        await asyncio.sleep(1)
        
        print("Log entries written (check /a0/tmp/chats/logs/)")
        await logger.stop()
    
    asyncio.run(test_logger())

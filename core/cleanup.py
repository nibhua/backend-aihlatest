import asyncio
import signal
import logging
import threading
import time
import os
from typing import Set
import weakref

logger = logging.getLogger(__name__)

class TaskManager:
    """Manages async tasks to ensure proper cleanup."""
    
    def __init__(self):
        self._tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
    
    def add_task(self, task: asyncio.Task) -> asyncio.Task:
        """Add a task to be managed."""
        self._tasks.add(task)
        task.add_done_callback(self._task_done)
        return task
    
    def _task_done(self, task: asyncio.Task) -> None:
        """Callback when a task is done."""
        try:
            self._tasks.discard(task)
            # Get the result to handle any exceptions
            task.result()
        except asyncio.CancelledError:
            logger.debug("Task was cancelled")
        except Exception as e:
            logger.error(f"Task failed with error: {e}")
    
    async def cancel_all_tasks(self) -> None:
        """Cancel all managed tasks."""
        if not self._tasks:
            return
        
        logger.info(f"Cancelling {len(self._tasks)} tasks")
        
        # Cancel all tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        # Wait for all tasks to complete with timeout
        if self._tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*self._tasks, return_exceptions=True), timeout=5.0)
                logger.info("All tasks cancelled successfully")
            except asyncio.TimeoutError:
                logger.warning("Some tasks did not complete within 5 second shutdown timeout - forcing shutdown")
                # Force cleanup of remaining tasks
                remaining_tasks = [task for task in self._tasks if not task.done()]
                if remaining_tasks:
                    logger.warning(f"Forcefully abandoning {len(remaining_tasks)} tasks")
                    self._tasks.clear()
    
    async def shutdown(self) -> None:
        """Shutdown the task manager."""
        self._shutdown_event.set()
        try:
            await asyncio.wait_for(self.cancel_all_tasks(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.error("Task manager shutdown timed out after 10 seconds - forcing exit")
        
        # Force cleanup of any remaining threads
        self._force_thread_cleanup()
    
    def _force_thread_cleanup(self) -> None:
        """Force cleanup of any remaining threads that might be hanging."""
        try:
            # Get all active threads
            active_threads = threading.enumerate()
            # Filter out main thread and daemon threads (they should exit automatically)
            non_daemon_threads = [t for t in active_threads if t != threading.main_thread() and not t.daemon]
            
            if non_daemon_threads:
                logger.warning(f"Found {len(non_daemon_threads)} non-daemon threads during shutdown")
                for thread in non_daemon_threads:
                    logger.warning(f"Thread: {thread.name} (alive: {thread.is_alive()})")
            
            # Give threads a short time to finish naturally
            time.sleep(0.5)
            
            # Force exit if any non-daemon threads are still running
            if any(t.is_alive() for t in non_daemon_threads):
                logger.error("Non-daemon threads still running after shutdown timeout - forcing exit")
        except Exception as e:
            logger.error(f"Error during thread cleanup: {e}")

# Global task manager instance
task_manager = TaskManager()

def setup_cleanup_handlers(app):
    """Setup cleanup handlers for the FastAPI app."""
    
    @app.on_event("startup")
    async def startup_event():
        logger.info("Starting up with task management")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Shutting down, cleaning up tasks...")
        try:
            await asyncio.wait_for(task_manager.shutdown(), timeout=15.0)
            logger.info("Cleanup complete")
        except asyncio.TimeoutError:
            logger.error("Application shutdown timed out after 15 seconds - forcing exit")
            # Force exit after timeout
            os._exit(1)

def handle_sigterm(signum, frame):
    """Handle SIGTERM signal for graceful shutdown."""
    logger.info("Received SIGTERM, initiating graceful shutdown...")
    try:
        # Get the current event loop
        loop = asyncio.get_event_loop()
        # Create a task with timeout
        task = asyncio.create_task(task_manager.shutdown())
        # Schedule the task to be cancelled after 15 seconds
        loop.call_later(15.0, task.cancel)
        # Force exit after 20 seconds if still running
        loop.call_later(20.0, lambda: os._exit(1))
    except RuntimeError:
        # No event loop running, just log and exit
        logger.info("No event loop running, exiting immediately")
        os._exit(0)

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

async def create_managed_task(coro):
    """Create a task that will be properly managed and cleaned up."""
    task = asyncio.create_task(coro)
    return task_manager.add_task(task)

def cleanup_old_tasks():
    """Clean up completed tasks to prevent memory leaks."""
    # Remove completed tasks
    completed_tasks = {task for task in task_manager._tasks if task.done()}
    task_manager._tasks -= completed_tasks
    
    if completed_tasks:
        logger.debug(f"Cleaned up {len(completed_tasks)} completed tasks") 
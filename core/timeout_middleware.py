import asyncio
import time
from typing import Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .error_handler import TimeoutError, create_error_response
from .logging_config import api_logger_instance, get_request_id

class TimeoutMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce timeout on all API endpoints."""
    
    def __init__(self, app: ASGIApp, timeout: float = 150.0):
        super().__init__(app)
        self.timeout = timeout
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID if not already present
        request_id = getattr(request.state, 'request_id', get_request_id())
        request.state.request_id = request_id
        
        # Start timing
        start_time = time.time()
        task = None
        
        try:
            # Create a task for the request processing
            task = asyncio.create_task(call_next(request))
            
            # Wait for the task to complete with timeout
            response = await asyncio.wait_for(task, timeout=self.timeout)
            
            # Log successful completion
            processing_time = time.time() - start_time
            api_logger_instance.log_performance(
                request_id=request_id,
                operation="request_complete",
                duration=processing_time,
                status_code=response.status_code
            )
            
            return response
            
        except asyncio.TimeoutError:
            # Handle timeout - ensure task is properly cancelled
            processing_time = time.time() - start_time
            
            # Cancel the task if it's still running
            if task and not task.done():
                task.cancel()
                try:
                    # Wait a short time for the task to actually cancel
                    await asyncio.wait_for(task, timeout=1.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    # Task is taking too long to cancel, log it but continue
                    api_logger_instance.log_error(
                        request_id=request_id,
                        error=Exception("Task cancellation timeout"),
                        status_code=500,
                        processing_time=processing_time
                    )
            
            # Log timeout
            api_logger_instance.log_error(
                request_id=request_id,
                error=TimeoutError(f"Request timed out after {self.timeout} seconds"),
                status_code=408,
                processing_time=processing_time
            )
            
            # Return timeout error response
            timeout_error = TimeoutError(f"Request timed out after {self.timeout} seconds")
            return create_error_response(timeout_error, request_id)
            
        except Exception as e:
            # Handle other exceptions
            processing_time = time.time() - start_time
            
            # Cancel the task if it's still running
            if task and not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=1.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
            
            # Log error
            api_logger_instance.log_error(
                request_id=request_id,
                error=e,
                status_code=500,
                processing_time=processing_time
            )
            
            # Re-raise the exception to be handled by other middleware
            raise 
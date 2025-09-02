"""
Concurrency control decorators for FastAPI endpoints.

These decorators provide easy-to-use concurrency control for any endpoint
without requiring manual slot management.
"""

import asyncio
import time
import functools
from typing import Callable, Any, Optional
from fastapi import HTTPException
import logging

from .concurrency_manager import ConcurrencySlot

logger = logging.getLogger(__name__)

def with_concurrency_control(
    service_name: str,
    max_concurrent: Optional[int] = None,
    timeout_seconds: Optional[int] = None,
    queue_full_status_code: int = 503
):
    """
    Decorator to add concurrency control to FastAPI endpoints.
    
    Args:
        service_name: Name of the service for concurrency tracking
        max_concurrent: Override max concurrent requests (optional)
        timeout_seconds: Override timeout in seconds (optional)
        queue_full_status_code: HTTP status code when queue is full
    
    Usage:
        @with_concurrency_control("query_processing")
        async def my_endpoint():
            # Your endpoint logic here
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request object if available
            request = None
            request_id = None
            
            # Try to find request object in args or kwargs
            for arg in args:
                if hasattr(arg, 'state') and hasattr(arg, 'method'):
                    request = arg
                    break
            
            if request:
                request_id = getattr(request.state, 'request_id', None)
            
            # Use concurrency slot context manager
            try:
                async with ConcurrencySlot(service_name, request_id):
                    return await func(*args, **kwargs)
            except RuntimeError as e:
                logger.warning(f"Concurrency slot acquisition failed for {service_name}: {e}")
                raise HTTPException(
                    status_code=queue_full_status_code,
                    detail=f"Service {service_name} is currently at capacity. Please try again later."
                )
            except Exception as e:
                logger.error(f"Error in concurrency-controlled endpoint {service_name}: {e}")
                raise
        
        return wrapper
    return decorator

def with_priority_concurrency_control(
    service_name: str,
    priority: int = 1,
    max_concurrent: Optional[int] = None,
    timeout_seconds: Optional[int] = None
):
    """
    Decorator for priority-based concurrency control.
    
    Higher priority requests get preference in the queue.
    
    Args:
        service_name: Name of the service
        priority: Priority level (higher = more important)
        max_concurrent: Override max concurrent requests
        timeout_seconds: Override timeout in seconds
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # For now, we'll use the standard concurrency control
            # Priority-based queuing can be implemented later
            return await with_concurrency_control(service_name, max_concurrent, timeout_seconds)(func)(*args, **kwargs)
        
        return wrapper
    return decorator

def with_rate_limiting(
    service_name: str,
    requests_per_minute: int = 60,
    burst_size: int = 10
):
    """
    Decorator for rate limiting (requests per time period).
    
    Args:
        service_name: Name of the service
        requests_per_minute: Maximum requests per minute
        burst_size: Maximum burst requests allowed
    """
    def decorator(func: Callable) -> Callable:
        # This is a placeholder for rate limiting implementation
        # Can be implemented with Redis or in-memory tracking
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # For now, just call the original function
            # Rate limiting logic would go here
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator

def with_circuit_breaker(
    service_name: str,
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    expected_exception: type = Exception
):
    """
    Decorator for circuit breaker pattern.
    
    Prevents cascading failures by temporarily stopping requests
    to a failing service.
    
    Args:
        service_name: Name of the service
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Time to wait before trying again
        expected_exception: Exception type to count as failures
    """
    def decorator(func: Callable) -> Callable:
        # This is a placeholder for circuit breaker implementation
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Circuit breaker logic would go here
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator

# Convenience decorators for common services
def podcast_concurrency(func: Callable) -> Callable:
    """Convenience decorator for podcast generation endpoints."""
    return with_concurrency_control("podcast_generation")(func)

def query_concurrency(func: Callable) -> Callable:
    """Convenience decorator for query processing endpoints."""
    return with_concurrency_control("query_processing")(func)

def build_concurrency(func: Callable) -> Callable:
    """Convenience decorator for collection build endpoints."""
    return with_concurrency_control("collection_build")(func)

def snippet_concurrency(func: Callable) -> Callable:
    """Convenience decorator for snippet processing endpoints."""
    return with_concurrency_control("snippet_processing")(func)

def chat_concurrency(func: Callable) -> Callable:
    """Convenience decorator for chat processing endpoints."""
    return with_concurrency_control("chat_processing")(func)

def summary_concurrency(func: Callable) -> Callable:
    """Convenience decorator for summary generation endpoints."""
    return with_concurrency_control("summary_generation")(func)

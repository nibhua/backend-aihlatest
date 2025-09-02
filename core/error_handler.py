import asyncio
import time
import threading
from typing import Dict, Any, Optional, Callable
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
import aiohttp
import google.generativeai as genai
from google.generativeai.types import StopCandidateException, BlockedPromptException
import requests
from requests.exceptions import Timeout, ConnectionError, RequestException
import logging

logger = logging.getLogger(__name__)

# Custom exception classes
class TimeoutError(Exception):
    """Raised when an operation times out."""
    pass

class RateLimitError(Exception):
    """Raised when API rate limit is exceeded."""
    pass

class NetworkError(Exception):
    """Raised when there's a network connectivity issue."""
    pass

class GeminiAPIError(Exception):
    """Raised when there's a specific Gemini API error."""
    pass

class ValidationError(Exception):
    """Raised when input validation fails."""
    pass

class ResourceNotFoundError(Exception):
    """Raised when a requested resource is not found."""
    pass

# Error mapping to HTTP status codes
ERROR_STATUS_MAPPING = {
    TimeoutError: 408,  # Request Timeout
    RateLimitError: 429,  # Too Many Requests
    NetworkError: 503,  # Service Unavailable
    GeminiAPIError: 502,  # Bad Gateway
    ValidationError: 400,  # Bad Request
    ResourceNotFoundError: 404,  # Not Found
    ValueError: 400,  # Bad Request
    KeyError: 400,  # Bad Request
    FileNotFoundError: 404,  # Not Found
    PermissionError: 403,  # Forbidden
}

# Gemini API specific error mapping
GEMINI_ERROR_MAPPING = {
    "quota_exceeded": 429,
    "rate_limit_exceeded": 429,
    "resource_exhausted": 429,
    "permission_denied": 403,
    "invalid_argument": 400,
    "not_found": 404,
    "unavailable": 503,
    "deadline_exceeded": 408,
    "internal": 500,
    "unknown": 500,
}

def get_http_status_code(error: Exception) -> int:
    """Get appropriate HTTP status code for an exception."""
    error_type = type(error)
    
    # Check custom error mapping first
    if error_type in ERROR_STATUS_MAPPING:
        return ERROR_STATUS_MAPPING[error_type]
    
    # Check for specific error messages
    error_message = str(error).lower()
    
    # Network/connection errors
    if any(keyword in error_message for keyword in ['connection', 'network', 'unreachable', 'dns']):
        return 503
    
    # Timeout errors
    if any(keyword in error_message for keyword in ['timeout', 'timed out', 'deadline']):
        return 408
    
    # Rate limiting
    if any(keyword in error_message for keyword in ['rate limit', 'quota', 'too many requests']):
        return 429
    
    # Default to 500 for unknown errors
    return 500

def create_error_response(error: Exception, request_id: Optional[str] = None) -> JSONResponse:
    """Create a standardized error response."""
    status_code = get_http_status_code(error)
    
    error_data = {
        "error": {
            "type": type(error).__name__,
            "message": str(error),
            "status_code": status_code,
            "timestamp": time.time()
        }
    }
    
    if request_id:
        error_data["error"]["request_id"] = request_id
    
    # Add specific error details based on type
    if isinstance(error, TimeoutError):
        error_data["error"]["details"] = "The operation timed out. Please try again."
    elif isinstance(error, RateLimitError):
        error_data["error"]["details"] = "API rate limit exceeded. Please wait before retrying."
    elif isinstance(error, NetworkError):
        error_data["error"]["details"] = "Network connectivity issue. Please check your connection."
    elif isinstance(error, GeminiAPIError):
        error_data["error"]["details"] = "External AI service error. Please try again later."
    
    return JSONResponse(
        status_code=status_code,
        content=error_data
    )

async def handle_gemini_api_call(
    api_call: Callable,
    *args,
    timeout: float = 150.0,
    max_retries: int = 3,
    **kwargs
) -> Any:
    """
    Handle Gemini API calls with timeout, retries, and proper error handling.
    
    Args:
        api_call: The API call function to execute
        timeout: Timeout in seconds (default 150)
        max_retries: Maximum number of retries
        *args, **kwargs: Arguments to pass to the API call
    
    Returns:
        API response
    
    Raises:
        TimeoutError: If the call times out
        RateLimitError: If rate limit is exceeded
        NetworkError: If there's a network issue
        GeminiAPIError: For other Gemini API errors
    """
    last_error = None
    
    for attempt in range(max_retries + 1):
        task = None
        try:
            # Create a task with timeout
            task = asyncio.create_task(api_call(*args, **kwargs))
            
            try:
                result = await asyncio.wait_for(task, timeout=timeout)
                return result
            except asyncio.TimeoutError:
                # Ensure task is properly cancelled
                if task and not task.done():
                    task.cancel()
                    try:
                        # Wait a short time for the task to actually cancel
                        await asyncio.wait_for(task, timeout=1.0)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        # Task is taking too long to cancel, log it but continue
                        logger.warning(f"Task cancellation timeout on attempt {attempt + 1}")
                raise TimeoutError(f"API call timed out after {timeout} seconds")
                
        except TimeoutError:
            last_error = TimeoutError(f"Timeout on attempt {attempt + 1}/{max_retries + 1}")
            if attempt == max_retries:
                raise last_error
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
        except asyncio.CancelledError:
            # Handle task cancellation
            last_error = TimeoutError(f"Task was cancelled on attempt {attempt + 1}")
            if attempt == max_retries:
                raise last_error
            await asyncio.sleep(2 ** attempt)
            
        except StopCandidateException as e:
            # Handle Gemini API specific errors
            error_message = str(e).lower()
            
            if any(keyword in error_message for keyword in ['quota', 'rate limit', 'resource exhausted']):
                raise RateLimitError(f"Gemini API rate limit exceeded: {str(e)}")
            elif any(keyword in error_message for keyword in ['permission', 'invalid']):
                raise GeminiAPIError(f"Gemini API permission error: {str(e)}")
            else:
                raise GeminiAPIError(f"Gemini API error: {str(e)}")
                
        except BlockedPromptException as e:
            raise ValidationError(f"Content blocked by safety filters: {str(e)}")
            
        except ConnectionError as e:
            last_error = NetworkError(f"Network connection error: {str(e)}")
            if attempt == max_retries:
                raise last_error
            await asyncio.sleep(2 ** attempt)
            
        except Exception as e:
            # Log unexpected errors
            logger.error(f"Unexpected error in API call (attempt {attempt + 1}): {str(e)}")
            last_error = e
            if attempt == max_retries:
                raise GeminiAPIError(f"Unexpected API error: {str(e)}")
            await asyncio.sleep(2 ** attempt)
        finally:
            # Ensure task is cleaned up if it's still running
            if task and not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=0.5)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
    
    # This should never be reached, but just in case
    raise last_error or GeminiAPIError("Unknown API error")

def handle_sync_api_call(
    api_call: Callable,
    *args,
    timeout: float = 150.0,
    max_retries: int = 3,
    **kwargs
) -> Any:
    """
    Handle synchronous API calls with timeout, retries, and proper error handling.
    
    Args:
        api_call: The API call function to execute
        timeout: Timeout in seconds (default 150)
        max_retries: Maximum number of retries
        *args, **kwargs: Arguments to pass to the API call
    
    Returns:
        API response
    
    Raises:
        TimeoutError: If the call times out
        RateLimitError: If rate limit is exceeded
        NetworkError: If there's a network issue
        GeminiAPIError: For other Gemini API errors
    """
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            # Use threading to prevent blocking the event loop
            result = None
            exception = None
            
            def run_api_call():
                nonlocal result, exception
                try:
                    result = api_call(*args, **kwargs)
                except Exception as e:
                    exception = e
            
            # Run the API call in a separate thread with timeout
            thread = threading.Thread(target=run_api_call)
            thread.daemon = True  # Don't block application shutdown
            thread.start()
            
            # Use a shorter timeout for shutdown scenarios
            join_timeout = min(timeout, 5.0)  # Max 5 seconds for thread join
            thread.join(timeout=join_timeout)
            
            if thread.is_alive():
                # Thread is still running, timeout occurred
                logger.warning(f"Thread timeout after {join_timeout} seconds, abandoning thread")
                raise TimeoutError(f"Request timed out after {join_timeout} seconds")
            
            if exception:
                raise exception
                
            return result
            
        except (Timeout, TimeoutError) as e:
            last_error = TimeoutError(f"Request timed out after {timeout} seconds: {str(e)}")
            if attempt == max_retries:
                raise last_error
            time.sleep(2 ** attempt)
            
        except ConnectionError as e:
            last_error = NetworkError(f"Network connection error: {str(e)}")
            if attempt == max_retries:
                raise last_error
            time.sleep(2 ** attempt)
            
        except RequestException as e:
            # Handle HTTP errors
            if hasattr(e, 'response') and e.response is not None:
                status_code = e.response.status_code
                if status_code == 429:
                    raise RateLimitError(f"Rate limit exceeded: {str(e)}")
                elif status_code >= 500:
                    last_error = NetworkError(f"Server error ({status_code}): {str(e)}")
                else:
                    last_error = GeminiAPIError(f"HTTP error ({status_code}): {str(e)}")
            else:
                last_error = NetworkError(f"Request error: {str(e)}")
                
            if attempt == max_retries:
                raise last_error
            time.sleep(2 ** attempt)
            
        except Exception as e:
            # Log unexpected errors
            logger.error(f"Unexpected error in sync API call (attempt {attempt + 1}): {str(e)}")
            last_error = e
            if attempt == max_retries:
                raise GeminiAPIError(f"Unexpected API error: {str(e)}")
            time.sleep(2 ** attempt)
    
    # This should never be reached, but just in case
    raise last_error or GeminiAPIError("Unknown API error")

def validate_api_key(api_key: str) -> bool:
    """Validate that an API key is present and properly formatted."""
    if not api_key or api_key.strip() == "":
        raise ValidationError("API key is required")
    
    # Basic validation for Gemini API key format
    if not api_key.startswith("AIza"):
        raise ValidationError("Invalid API key format")
    
    return True

def check_network_connectivity() -> bool:
    """Check if there's basic network connectivity."""
    try:
        # Try to connect to a reliable service
        response = requests.get("https://www.google.com", timeout=5)
        return response.status_code == 200
    except Exception:
        return False 
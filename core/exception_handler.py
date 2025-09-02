from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import traceback
import time

from .error_handler import (
    TimeoutError, RateLimitError, NetworkError, GeminiAPIError,
    ValidationError, ResourceNotFoundError, create_error_response,
    get_http_status_code
)
from .logging_config import api_logger_instance, get_request_id

async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Global exception handler that catches all exceptions and returns appropriate HTTP responses.
    """
    # Get request ID
    request_id = getattr(request.state, 'request_id', get_request_id())
    
    # Log the exception
    api_logger_instance.log_error(
        request_id=request_id,
        error=exc,
        status_code=get_http_status_code(exc),
        traceback=traceback.format_exc()
    )
    
    # Handle specific exception types
    if isinstance(exc, TimeoutError):
        return create_error_response(exc, request_id)
    
    elif isinstance(exc, RateLimitError):
        return create_error_response(exc, request_id)
    
    elif isinstance(exc, NetworkError):
        return create_error_response(exc, request_id)
    
    elif isinstance(exc, GeminiAPIError):
        return create_error_response(exc, request_id)
    
    elif isinstance(exc, ValidationError):
        return create_error_response(exc, request_id)
    
    elif isinstance(exc, ResourceNotFoundError):
        return create_error_response(exc, request_id)
    
    elif isinstance(exc, RequestValidationError):
        # Handle FastAPI validation errors
        error_data = {
            "error": {
                "type": "ValidationError",
                "message": "Request validation failed",
                "status_code": 422,
                "timestamp": time.time(),
                "request_id": request_id,
                "details": "Invalid request data format",
                "validation_errors": exc.errors()
            }
        }
        return JSONResponse(status_code=422, content=error_data)
    
    elif isinstance(exc, StarletteHTTPException):
        # Handle Starlette HTTP exceptions
        error_data = {
            "error": {
                "type": "HTTPException",
                "message": exc.detail,
                "status_code": exc.status_code,
                "timestamp": time.time(),
                "request_id": request_id,
                "details": "HTTP error occurred"
            }
        }
        return JSONResponse(status_code=exc.status_code, content=error_data)
    
    elif isinstance(exc, ValueError):
        # Handle ValueError (bad request)
        return create_error_response(exc, request_id)
    
    elif isinstance(exc, KeyError):
        # Handle KeyError (bad request)
        return create_error_response(exc, request_id)
    
    elif isinstance(exc, FileNotFoundError):
        # Handle FileNotFoundError (not found)
        return create_error_response(exc, request_id)
    
    elif isinstance(exc, PermissionError):
        # Handle PermissionError (forbidden)
        return create_error_response(exc, request_id)
    
    else:
        # Handle any other unexpected exceptions
        error_data = {
            "error": {
                "type": "InternalServerError",
                "message": "An unexpected error occurred",
                "status_code": 500,
                "timestamp": time.time(),
                "request_id": request_id,
                "details": "Please try again later or contact support if the problem persists"
            }
        }
        
        # In development, you might want to include the actual error message
        # In production, you should not expose internal error details
        if hasattr(request.app.state, 'debug') and request.app.state.debug:
            error_data["error"]["internal_error"] = str(exc)
            error_data["error"]["traceback"] = traceback.format_exc()
        
        return JSONResponse(status_code=500, content=error_data)

def setup_exception_handlers(app):
    """Setup global exception handlers for the FastAPI app."""
    
    # Register the global exception handler
    app.add_exception_handler(Exception, global_exception_handler)
    
    # Register specific exception handlers for better control
    app.add_exception_handler(TimeoutError, global_exception_handler)
    app.add_exception_handler(RateLimitError, global_exception_handler)
    app.add_exception_handler(NetworkError, global_exception_handler)
    app.add_exception_handler(GeminiAPIError, global_exception_handler)
    app.add_exception_handler(ValidationError, global_exception_handler)
    app.add_exception_handler(ResourceNotFoundError, global_exception_handler)
    app.add_exception_handler(ValueError, global_exception_handler)
    app.add_exception_handler(KeyError, global_exception_handler)
    app.add_exception_handler(FileNotFoundError, global_exception_handler)
    app.add_exception_handler(PermissionError, global_exception_handler) 
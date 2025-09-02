import time
import json
from typing import Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse, FileResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .logging_config import api_logger_instance, get_request_id, sanitize_sensitive_data

class APILoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive API request/response logging."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate unique request ID
        request_id = get_request_id()
        
        # Add request ID to request state for use in endpoints
        request.state.request_id = request_id
        
        # Start timing
        start_time = time.time()
        
        # Extract basic request information without parsing form data
        request_data = {
            "content_type": request.headers.get("content-type", "unknown"),
            "content_length": request.headers.get("content-length", "unknown")
        }
        
        # Log the incoming request
        api_logger_instance.log_request(
            request=request,
            request_id=request_id,
            **request_data
        )
        
        try:
            # Process the request
            response = await call_next(request)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Extract basic response information without consuming the body
            response_data = {
                "type": "response",
                "status_code": response.status_code,
                "content_type": response.headers.get("content-type", "unknown"),
                "content_length": response.headers.get("content-length", "unknown")
            }
            
            # Add specific info based on response type
            if isinstance(response, FileResponse):
                response_data.update({
                    "type": "file_response",
                    "filename": getattr(response, 'filename', 'unknown'),
                    "media_type": getattr(response, 'media_type', 'unknown')
                })
            elif isinstance(response, JSONResponse):
                response_data.update({
                    "type": "json_response"
                })
            
            # Log the response
            api_logger_instance.log_response(
                request_id=request_id,
                response_data=response_data,
                status_code=response.status_code,
                processing_time=processing_time
            )
            
            return response
            
        except Exception as e:
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Log the error
            api_logger_instance.log_error(
                request_id=request_id,
                error=e,
                status_code=500,
                processing_time=processing_time
            )
            
            # Re-raise the exception
            raise 
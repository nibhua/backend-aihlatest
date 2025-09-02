import logging
import json
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import Request, Response
from fastapi.responses import JSONResponse
import traceback
import os
from pathlib import Path

# Create logs directory if it doesn't exist
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Configure logging
def setup_logging():
    """Setup comprehensive logging configuration for the application."""
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create handlers
    # File handler for all logs
    all_handler = logging.FileHandler(LOGS_DIR / "all.log")
    all_handler.setLevel(logging.DEBUG)
    all_handler.setFormatter(detailed_formatter)
    
    # File handler for API logs specifically
    api_handler = logging.FileHandler(LOGS_DIR / "api.log")
    api_handler.setLevel(logging.INFO)
    api_handler.setFormatter(detailed_formatter)
    
    # File handler for errors
    error_handler = logging.FileHandler(LOGS_DIR / "errors.log")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    
    # Console handler - only log to console in development
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(detailed_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(all_handler)
    root_logger.addHandler(api_handler)
    root_logger.addHandler(error_handler)
    
    # Only add console handler in development mode
    if os.getenv("ENVIRONMENT", "development") == "development":
        root_logger.addHandler(console_handler)
    
    # Create API logger
    api_logger = logging.getLogger("api")
    api_logger.setLevel(logging.INFO)
    
    return api_logger

# Initialize the API logger
api_logger = setup_logging()

class APILogger:
    """Comprehensive API logging utility for request/response tracking."""
    
    def __init__(self):
        self.logger = api_logger
    
    def log_request(self, request: Request, request_id: str, **kwargs):
        """Log incoming request details."""
        try:
            # Extract request information
            request_data = {
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "headers": dict(request.headers),
                "client_ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
                "timestamp": datetime.now().isoformat(),
                **kwargs
            }
            
            # Try to get request body for POST/PUT requests
            if request.method in ["POST", "PUT", "PATCH"]:
                try:
                    # For form data, we'll log the form fields
                    if "application/x-www-form-urlencoded" in request.headers.get("content-type", ""):
                        request_data["content_type"] = "form_data"
                    elif "multipart/form-data" in request.headers.get("content-type", ""):
                        request_data["content_type"] = "multipart_form_data"
                    elif "application/json" in request.headers.get("content-type", ""):
                        request_data["content_type"] = "json"
                    else:
                        request_data["content_type"] = "unknown"
                except Exception as e:
                    request_data["body_parsing_error"] = str(e)
            
            self.logger.info(f"REQUEST [{request_id}]: {json.dumps(request_data, indent=2)}")
            
        except Exception as e:
            self.logger.error(f"Error logging request: {str(e)}")
    
    def log_response(self, request_id: str, response_data: Any, status_code: int, 
                    processing_time: float, **kwargs):
        """Log response details."""
        try:
            response_log = {
                "request_id": request_id,
                "status_code": status_code,
                "processing_time_ms": round(processing_time * 1000, 2),
                "timestamp": datetime.now().isoformat(),
                **kwargs
            }
            
            # Handle different response types
            if isinstance(response_data, dict):
                # For our new response format
                response_log["response_type"] = response_data.get("type", "unknown")
                response_log["response_data"] = response_data
            elif isinstance(response_data, str):
                # For string responses, log first 500 chars
                response_log["response_type"] = "text"
                response_log["response_data"] = response_data[:500] + "..." if len(response_data) > 500 else response_data
            elif hasattr(response_data, 'body'):
                # For file responses
                response_log["response_type"] = "file"
                response_log["file_info"] = {
                    "filename": getattr(response_data, 'filename', 'unknown'),
                    "media_type": getattr(response_data, 'media_type', 'unknown')
                }
            else:
                response_log["response_type"] = "other"
                response_log["response_data"] = str(response_data)[:500]
            
            self.logger.info(f"RESPONSE [{request_id}]: {json.dumps(response_log, indent=2)}")
            
        except Exception as e:
            self.logger.error(f"Error logging response: {str(e)}")
    
    def log_error(self, request_id: str, error: Exception, status_code: int = 500, **kwargs):
        """Log error details."""
        try:
            error_log = {
                "request_id": request_id,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "status_code": status_code,
                "timestamp": datetime.now().isoformat(),
                "traceback": traceback.format_exc(),
                **kwargs
            }
            
            self.logger.error(f"ERROR [{request_id}]: {json.dumps(error_log, indent=2)}")
            
        except Exception as e:
            self.logger.error(f"Error logging error: {str(e)}")
    
    def log_performance(self, request_id: str, operation: str, duration: float, **kwargs):
        """Log performance metrics."""
        try:
            perf_log = {
                "request_id": request_id,
                "operation": operation,
                "duration_ms": round(duration * 1000, 2),
                "timestamp": datetime.now().isoformat(),
                **kwargs
            }
            
            self.logger.info(f"PERFORMANCE [{request_id}]: {json.dumps(perf_log, indent=2)}")
            
        except Exception as e:
            self.logger.error(f"Error logging performance: {str(e)}")

    def log_info(self, data: Any, request_id: Optional[str] = None, **kwargs):
        """Generic info logger for structured data used across services.

        Many modules call api_logger_instance.log_info(payload). This method
        ensures backward compatibility and logs a JSON representation of the
        provided data. An optional request_id may be provided to correlate logs.
        """
        try:
            info_log = {
                "request_id": request_id or "",
                "timestamp": datetime.now().isoformat(),
                "data": data,
                **kwargs
            }
            self.logger.info(f"INFO: {json.dumps(info_log, indent=2, default=str)}")
        except Exception as e:
            self.logger.error(f"Error logging info: {str(e)}")

# Global API logger instance
api_logger_instance = APILogger()

def get_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())

def sanitize_sensitive_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove sensitive information from logged data."""
    sensitive_keys = ['password', 'token', 'api_key', 'secret', 'authorization']
    sanitized = data.copy()
    
    for key in sensitive_keys:
        if key in sanitized:
            sanitized[key] = '[REDACTED]'
    
    return sanitized 
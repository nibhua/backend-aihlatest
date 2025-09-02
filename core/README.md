# core/

Core utilities, middleware, and shared components for the AIH backend.

## What it does

- **Logging**: Centralized API logging with performance tracking and request ID management
- **Error Handling**: Custom exception classes and error response utilities
- **Middleware**: Request/response middleware for logging and timeout handling
- **Task Management**: Background task creation and cleanup utilities
- **API Utilities**: Common functions for API key validation and network checks

## Files

- `logging_config.py` — API logging configuration and performance tracking
- `error_handler.py` — Custom exceptions and error handling utilities
- `exception_handler.py` — Global exception handling setup
- `middleware.py` — Request/response logging middleware
- `timeout_middleware.py` — Request timeout handling
- `cleanup.py` — Background task management and cleanup
- `json_utils.py` — JSON utility functions
- `__init__.py` — Module exports and re-exports

## Key Components

- **Custom Exceptions**: TimeoutError, RateLimitError, NetworkError, etc.
- **Logging**: Structured logging with request IDs and performance metrics
- **Middleware**: APILoggingMiddleware, TimeoutMiddleware
- **Task Management**: Background task creation and cleanup

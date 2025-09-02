# chat_service/

Manages chat sessions and LLM interactions for document-based conversations.

## What it does

- **Session Management**: Creates and maintains chat sessions with optional document collection context
- **LLM Integration**: Handles message exchange with language models
- **Context Awareness**: Incorporates relevant document sections into conversations
- **Performance Logging**: Tracks chat performance metrics and response times

## Files

- `chat_manager.py` — Core chat session management and LLM interaction logic
- `router.py` — FastAPI endpoints for chat operations
- `__init__.py` — Module exports

## Endpoints

- `POST /chat/sessions` — Create new chat session
- `POST /chat/sessions/{session_id}/chat` — Send message to LLM
- `GET /chat/sessions/{session_id}` — Get session information
- `DELETE /chat/sessions/{session_id}` — End chat session

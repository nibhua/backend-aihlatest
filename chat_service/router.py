from fastapi import APIRouter, HTTPException, Form, Request
from typing import List, Dict, Any, Optional
import json
import time
from .chat_manager import chat_manager

# Import logging components
from core import api_logger_instance, get_request_id
from core.concurrency_decorators import chat_concurrency

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/sessions")
async def create_chat_session(request: Request, collection_id: Optional[str] = Form(default=None)):
    """
    Create a new chat session.
    
    Args:
        collection_id: Optional collection ID to provide document context
    
    Returns:
        New session information
    """
    start_time = time.time()
    request_id = getattr(request.state, 'request_id', get_request_id())
    
    # Log session creation start
    api_logger_instance.log_performance(
        request_id=request_id,
        operation="chat_session_create_start",
        duration=0,
        has_collection_id=bool(collection_id)
    )
    
    try:
        session_id = chat_manager.create_session(collection_id)
        
        processing_time = time.time() - start_time
        
        # Log session creation completion
        api_logger_instance.log_performance(
            request_id=request_id,
            operation="chat_session_create_complete",
            duration=processing_time,
            session_id=session_id
        )
        
        response_data = {
            "session_id": session_id,
            "collection_id": collection_id,
            "message": "Chat session created successfully"
        }
        
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating session: {str(e)}")

@router.post("/sessions/{session_id}/chat")
@chat_concurrency
async def chat_with_llm(
    request: Request,
    session_id: str,
    message: str = Form(...),
    context_sections: Optional[str] = Form(default=None)  # JSON string of relevant sections
):
    """
    Send a message to the LLM and get a response.
    
    Args:
        session_id: The chat session ID
        message: User's message
        context_sections: Optional JSON string of relevant document sections
    
    Returns:
        LLM response and session information
    """
    start_time = time.time()
    request_id = getattr(request.state, 'request_id', get_request_id())
    
    # Log chat message start
    api_logger_instance.log_performance(
        request_id=request_id,
        operation="chat_message_start",
        duration=0,
        session_id=session_id,
        message_length=len(message),
        has_context_sections=bool(context_sections)
    )
    
    try:
        # Validate message
        if not message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Parse context sections if provided
        sections_data = []
        if context_sections:
            try:
                sections_data = json.loads(context_sections)
                if not isinstance(sections_data, list):
                    raise HTTPException(status_code=400, detail="context_sections must be a JSON array")
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON in context_sections")
        
        # Process chat message
        response_data = chat_manager.chat(
            session_id=session_id,
            message=message,
            context_sections=sections_data
        )
        
        processing_time = time.time() - start_time
        
        # Log chat message completion
        api_logger_instance.log_performance(
            request_id=request_id,
            operation="chat_message_complete",
            duration=processing_time,
            session_id=session_id,
            response_length=len(response_data.get('response', ''))
        )
        
        return response_data
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@router.get("/sessions/{session_id}/history")
async def get_chat_history(session_id: str, limit: Optional[int] = None):
    """
    Get the conversation history for a session.
    
    Args:
        session_id: The chat session ID
        limit: Optional limit on number of messages to return
    
    Returns:
        Conversation history
    """
    try:
        history = chat_manager.get_session_history(session_id, limit)
        return {
            "session_id": session_id,
            "messages": history,
            "total_messages": len(history)
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving history: {str(e)}")

@router.get("/sessions")
async def list_chat_sessions():
    """
    List all chat sessions.
    
    Returns:
        List of session information
    """
    try:
        sessions = chat_manager.list_sessions()
        return {
            "sessions": sessions,
            "total_sessions": len(sessions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing sessions: {str(e)}")

@router.delete("/sessions/{session_id}")
async def delete_chat_session(session_id: str):
    """
    Delete a chat session.
    
    Args:
        session_id: The chat session ID to delete
    
    Returns:
        Deletion confirmation
    """
    try:
        success = chat_manager.delete_session(session_id)
        if success:
            return {"message": f"Session {session_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting session: {str(e)}")

@router.post("/sessions/{session_id}/context")
async def update_session_context(
    session_id: str,
    context_sections: str = Form(...)  # JSON string of document sections
):
    """
    Update the document context for a chat session.
    
    Args:
        session_id: The chat session ID
        context_sections: JSON string of document sections to add to context
    
    Returns:
        Update confirmation
    """
    try:
        # Parse context sections
        try:
            sections_data = json.loads(context_sections)
            if not isinstance(sections_data, list):
                raise HTTPException(status_code=400, detail="context_sections must be a JSON array")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in context_sections")
        
        # Get session
        session = chat_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Update context
        session.set_document_context(sections_data)
        chat_manager._save_session(session)
        
        return {
            "message": "Session context updated successfully",
            "context_documents_count": len(sections_data)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating context: {str(e)}")

@router.get("/health")
async def chat_health_check():
    """Health check for chat service."""
    return {
        "status": "healthy",
        "active_sessions": len(chat_manager.sessions),
        "model": chat_manager.model
    }


import os
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv():
        return None
import json
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path
import google.generativeai as genai
from datetime import datetime

# Load environment variables from .env and configure Gemini if present
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

class ChatSession:
    """Represents a chat session with conversation history."""
    
    def __init__(self, session_id: str, collection_id: Optional[str] = None):
        self.session_id = session_id
        self.collection_id = collection_id
        self.messages: List[Dict[str, str]] = []
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        self.context_documents: List[Dict[str, Any]] = []
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a message to the conversation history."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        if metadata:
            message["metadata"] = metadata
        
        self.messages.append(message)
        self.updated_at = datetime.now().isoformat()
    
    def get_conversation_history(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """Get conversation history formatted for OpenAI API."""
        messages = []
        
        # Add system message with context
        system_content = self._build_system_message()
        messages.append({"role": "system", "content": system_content})
        
        # Add conversation history
        history = self.messages[-limit:] if limit else self.messages
        for msg in history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        return messages
    
    def _build_system_message(self) -> str:
        """Build system message with document context."""
        base_message = """You are an AI assistant helping users understand and analyze their document collection. 
        You have access to the user's personal document library and can provide insights based on their uploaded content.
        
        Guidelines:
        - Only reference information from the user's uploaded documents
        - Be helpful and provide detailed explanations when asked
        - If you don't have relevant information in the user's documents, say so clearly
        - Cite specific documents when referencing information
        - Help users discover connections and insights across their document collection"""
        
        if self.context_documents:
            context_info = "\n\nAvailable documents in your collection:\n"
            for doc in self.context_documents[:10]:  # Limit to prevent token overflow
                doc_name = doc.get("file_name", "Unknown")
                doc_summary = doc.get("summary", "No summary available")
                context_info += f"- {doc_name}: {doc_summary}\n"
            
            base_message += context_info
        
        return base_message
    
    def set_document_context(self, documents: List[Dict[str, Any]]):
        """Set the document context for this session."""
        self.context_documents = documents
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "collection_id": self.collection_id,
            "messages": self.messages,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "context_documents": self.context_documents
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatSession':
        """Create session from dictionary."""
        session = cls(data["session_id"], data.get("collection_id"))
        session.messages = data.get("messages", [])
        session.created_at = data.get("created_at", datetime.now().isoformat())
        session.updated_at = data.get("updated_at", datetime.now().isoformat())
        session.context_documents = data.get("context_documents", [])
        return session

class ChatManager:
    """Manages chat sessions and LLM interactions."""
    
    def __init__(self):
        # Allow model override via environment (GEMINI_MODEL or LLM_MODEL)
        self.model = os.getenv("GEMINI_MODEL") or os.getenv("LLM_MODEL") or "gemini-2.0-flash-lite"
        self.sessions: Dict[str, ChatSession] = {}
        self.legacy_sessions_dir = Path("chat_service/sessions")
        self.legacy_sessions_dir.mkdir(parents=True, exist_ok=True)
        self._load_sessions()
    
    def _get_sessions_dir(self, collection_id: Optional[str] = None) -> Path:
        """Get the sessions directory for a collection."""
        if collection_id and collection_id.startswith('col_'):
            try:
                from core.workspace_manager import workspace_manager
                return workspace_manager.get_chat_sessions_path(collection_id)
            except Exception:
                pass  # Fall back to legacy path
        
        # Fall back to legacy path
        return self.legacy_sessions_dir
    
    def create_session(self, collection_id: Optional[str] = None) -> str:
        """Create a new chat session."""
        session_id = str(uuid.uuid4())
        session = ChatSession(session_id, collection_id)
        
        # Load document context if collection_id is provided
        if collection_id:
            documents = self._load_collection_documents(collection_id)
            session.set_document_context(documents)
        
        self.sessions[session_id] = session
        self._save_session(session)
        return session_id
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get an existing chat session."""
        return self.sessions.get(session_id)
    
    def chat(self, session_id: str, message: str, context_sections: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Process a chat message and return the response.
        
        Args:
            session_id: The chat session ID
            message: User's message
            context_sections: Optional relevant sections to include in context
        
        Returns:
            Dictionary with response and metadata
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Add user message to history
        session.add_message("user", message)
        
        # Prepare context if sections provided
        enhanced_message = message
        if context_sections:
            context_text = self._format_context_sections(context_sections)
            enhanced_message = f"{message}\n\nRelevant context from your documents:\n{context_text}"
        
        try:
            # Get conversation history
            conversation = session.get_conversation_history(limit=20)  # Limit to recent messages
            
            # Update the last user message with enhanced context
            if conversation and conversation[-1]["role"] == "user":
                conversation[-1]["content"] = enhanced_message
            
            # Convert conversation to Gemini format and generate response
            gemini_conversation = self._convert_to_gemini_format(conversation)
            
            model = genai.GenerativeModel(self.model)
            response = model.generate_content(
                gemini_conversation,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=1000,
                )
            )
            
            assistant_message = response.text
            
            # Log LLM output for debugging
            print(f"\n=== LLM OUTPUT LOG ===")
            print(f"Session ID: {session_id}")
            print(f"User Message: {message}")
            print(f"Enhanced Message: {enhanced_message}")
            print(f"LLM Response: {assistant_message}")
            print(f"Model: {self.model}")
            print(f"Context Sections Count: {len(context_sections) if context_sections else 0}")
            print(f"Timestamp: {datetime.now().isoformat()}")
            print(f"=== END LLM OUTPUT LOG ===\n")
            
            # Add assistant response to history
            session.add_message("assistant", assistant_message, {
                "model": self.model,
                "context_sections_count": len(context_sections) if context_sections else 0
            })
            
            # Save session
            self._save_session(session)
            
            return {
                "response": assistant_message,
                "session_id": session_id,
                "message_count": len(session.messages),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            # Add error to session for debugging
            session.add_message("system", f"Error: {str(e)}", {"error": True})
            self._save_session(session)
            raise e
    
    def get_session_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get the conversation history for a session."""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        messages = session.messages[-limit:] if limit else session.messages
        return messages
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all chat sessions."""
        sessions_info = []
        for session in self.sessions.values():
            last_message = session.messages[-1] if session.messages else None
            sessions_info.append({
                "session_id": session.session_id,
                "collection_id": session.collection_id,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "message_count": len(session.messages),
                "last_message_preview": last_message["content"][:100] + "..." if last_message else None
            })
        
        # Sort by updated_at descending
        sessions_info.sort(key=lambda x: x["updated_at"], reverse=True)
        return sessions_info
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a chat session."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            del self.sessions[session_id]
            
            # Delete session file from appropriate directory
            sessions_dir = self._get_sessions_dir(session.collection_id)
            session_file = sessions_dir / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()
            
            return True
        return False
    
    def _format_context_sections(self, sections: List[Dict[str, Any]]) -> str:
        """Format context sections for inclusion in chat."""
        context_parts = []
        
        for section in sections[:5]:  # Limit to top 5 sections
            doc_name = section.get("file_name", "Unknown Document")
            heading = section.get("heading", "")
            snippet = section.get("snippet", "")
            
            if heading:
                context_parts.append(f"From '{doc_name}' - {heading}:\n{snippet}")
            else:
                context_parts.append(f"From '{doc_name}':\n{snippet}")
        
        return "\n\n".join(context_parts)
    
    def _load_collection_documents(self, collection_id: str) -> List[Dict[str, Any]]:
        """Load document summaries for a collection."""
        # Try workspace-based path first if collection_id starts with 'col_'
        if collection_id.startswith('col_'):
            try:
                from core.workspace_manager import workspace_manager
                vector_dir = workspace_manager.get_vector_store_path(collection_id)
                meta_file = vector_dir / "meta.json"
                
                if meta_file.exists():
                    try:
                        with meta_file.open("r", encoding="utf-8") as f:
                            meta_data = json.load(f)
                            return meta_data.get("documents", [])
                    except Exception:
                        pass
            except Exception:
                pass  # Fall back to legacy path
        
        # Fall back to legacy path
        vector_dir = Path("vector_store") / collection_id
        meta_file = vector_dir / "meta.json"
        
        if meta_file.exists():
            try:
                with meta_file.open("r", encoding="utf-8") as f:
                    meta_data = json.load(f)
                    return meta_data.get("documents", [])
            except Exception:
                pass
        
        return []
    
    def _save_session(self, session: ChatSession):
        """Save session to disk."""
        sessions_dir = self._get_sessions_dir(session.collection_id)
        session_file = sessions_dir / f"{session.session_id}.json"
        try:
            with session_file.open("w", encoding="utf-8") as f:
                json.dump(session.to_dict(), f, indent=2)
        except Exception as e:
            print(f"Error saving session {session.session_id}: {e}")
    
    def _load_sessions(self):
        """Load existing sessions from disk."""
        # Load sessions from legacy directory
        if self.legacy_sessions_dir.exists():
            for session_file in self.legacy_sessions_dir.glob("*.json"):
                try:
                    with session_file.open("r", encoding="utf-8") as f:
                        session_data = json.load(f)
                        session = ChatSession.from_dict(session_data)
                        self.sessions[session.session_id] = session
                except Exception as e:
                    print(f"Error loading session from {session_file}: {e}")
        
        # Load sessions from all workspace directories
        try:
            from core.workspace_manager import workspace_manager
            for workspace_dir in workspace_manager.base_workspace_dir.iterdir():
                if not workspace_dir.is_dir():
                    continue
                
                sessions_dir = workspace_dir / "chat_sessions"
                if sessions_dir.exists():
                    for session_file in sessions_dir.glob("*.json"):
                        try:
                            with session_file.open("r", encoding="utf-8") as f:
                                session_data = json.load(f)
                                session = ChatSession.from_dict(session_data)
                                self.sessions[session.session_id] = session
                        except Exception as e:
                            print(f"Error loading session from {session_file}: {e}")
        except Exception:
            pass  # Workspace manager not available

    def _convert_to_gemini_format(self, conversation: List[Dict[str, str]]) -> str:
        """Convert OpenAI-style conversation to Gemini format."""
        formatted_parts = []
        
        for msg in conversation:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "system":
                # System messages become part of the context
                formatted_parts.append(f"System Instructions: {content}")
            elif role == "user":
                formatted_parts.append(f"User: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}")
        
        return "\n\n".join(formatted_parts)


# Global instance
chat_manager = ChatManager()


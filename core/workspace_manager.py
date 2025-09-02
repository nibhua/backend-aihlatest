"""
Workspace Manager for Isolated Collection Processing

This module provides utilities to manage isolated workspaces for each collection,
ensuring concurrent processing without conflicts.
"""

import os
import shutil
import time
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class WorkspaceManager:
    """
    Manages isolated workspaces for each collection ID.
    
    Each workspace contains:
    - uploads/     - uploaded files for this collection
    - vector_store/ - vector embeddings and metadata
    - summaries/   - generated summaries
    """
    
    def __init__(self, base_workspace_dir: Optional[str] = None):
        """
        Initialize workspace manager.
        
        Args:
            base_workspace_dir: Base directory for all workspaces. 
                               Defaults to 'workspaces' in backend root.
        """
        if base_workspace_dir is None:
            # Default to workspaces/ in backend root
            backend_root = Path(__file__).resolve().parents[1]
            base_workspace_dir = backend_root / "workspaces"
        
        self.base_workspace_dir = Path(base_workspace_dir)
        self.base_workspace_dir.mkdir(parents=True, exist_ok=True)
        
        # Collection retention time (2 hours by default)
        self.retention_hours = int(os.getenv("COLLECTION_RETENTION_HOURS", "2"))
        self.retention_seconds = self.retention_hours * 3600
        
        logger.info(f"WorkspaceManager initialized with base_dir: {self.base_workspace_dir}")
        logger.info(f"Collection retention: {self.retention_hours} hours")
    
    def get_workspace_path(self, collection_id: str) -> Path:
        """
        Get the workspace path for a collection ID.
        
        Args:
            collection_id: The collection identifier
            
        Returns:
            Path to the workspace directory
        """
        return self.base_workspace_dir / collection_id
    
    def get_uploads_path(self, collection_id: str) -> Path:
        """Get uploads directory path for a collection."""
        return self.get_workspace_path(collection_id) / "uploads"
    
    def get_vector_store_path(self, collection_id: str) -> Path:
        """Get vector store directory path for a collection."""
        return self.get_workspace_path(collection_id) / "vector_store"
    
    def get_summaries_path(self, collection_id: str) -> Path:
        """Get summaries directory path for a collection."""
        return self.get_workspace_path(collection_id) / "summaries"
    
    def get_audio_output_path(self, collection_id: str) -> Path:
        """Get audio output directory path for a collection."""
        return self.get_workspace_path(collection_id) / "audio_output"
    
    def get_chat_sessions_path(self, collection_id: str) -> Path:
        """Get chat sessions directory path for a collection."""
        return self.get_workspace_path(collection_id) / "chat_sessions"
    
    def create_workspace(self, collection_id: str) -> Path:
        """
        Create a new workspace for a collection.
        
        Args:
            collection_id: The collection identifier
            
        Returns:
            Path to the created workspace
        """
        workspace_path = self.get_workspace_path(collection_id)
        
        # Create all subdirectories
        self.get_uploads_path(collection_id).mkdir(parents=True, exist_ok=True)
        self.get_vector_store_path(collection_id).mkdir(parents=True, exist_ok=True)
        self.get_summaries_path(collection_id).mkdir(parents=True, exist_ok=True)
        self.get_audio_output_path(collection_id).mkdir(parents=True, exist_ok=True)
        self.get_chat_sessions_path(collection_id).mkdir(parents=True, exist_ok=True)
        
        # Create a metadata file with creation time
        metadata = {
            "collection_id": collection_id,
            "created_at": time.time(),
            "created_at_iso": time.strftime("%Y-%m-%d %H:%M:%S"),
            "retention_hours": self.retention_hours
        }
        
        metadata_path = workspace_path / "metadata.json"
        import json
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        
        logger.info(f"Created workspace for collection {collection_id} at {workspace_path}")
        return workspace_path
    
    def workspace_exists(self, collection_id: str) -> bool:
        """Check if a workspace exists for a collection."""
        return self.get_workspace_path(collection_id).exists()
    
    def get_workspace_metadata(self, collection_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a workspace."""
        metadata_path = self.get_workspace_path(collection_id) / "metadata.json"
        if not metadata_path.exists():
            return None
        
        try:
            import json
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"Error reading metadata for {collection_id}: {e}")
            return None
    
    def is_workspace_expired(self, collection_id: str) -> bool:
        """Check if a workspace has expired based on retention policy."""
        metadata = self.get_workspace_metadata(collection_id)
        if not metadata:
            return True  # If no metadata, consider expired
        
        created_at = metadata.get("created_at", 0)
        current_time = time.time()
        age_seconds = current_time - created_at
        
        return age_seconds > self.retention_seconds
    
    def cleanup_workspace(self, collection_id: str) -> bool:
        """
        Clean up a workspace (delete all files).
        
        Args:
            collection_id: The collection identifier
            
        Returns:
            True if cleanup was successful, False otherwise
        """
        workspace_path = self.get_workspace_path(collection_id)
        
        if not workspace_path.exists():
            return True  # Already clean
        
        try:
            shutil.rmtree(workspace_path)
            logger.info(f"Cleaned up workspace for collection {collection_id}")
            return True
        except Exception as e:
            logger.error(f"Error cleaning up workspace for {collection_id}: {e}")
            return False
    
    def cleanup_expired_workspaces(self) -> int:
        """
        Clean up all expired workspaces.
        
        Returns:
            Number of workspaces cleaned up
        """
        cleaned_count = 0
        
        if not self.base_workspace_dir.exists():
            return 0
        
        for workspace_dir in self.base_workspace_dir.iterdir():
            if not workspace_dir.is_dir():
                continue
            
            collection_id = workspace_dir.name
            if self.is_workspace_expired(collection_id):
                if self.cleanup_workspace(collection_id):
                    cleaned_count += 1
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} expired workspaces")
        
        return cleaned_count
    
    def cleanup_all_workspaces(self) -> int:
        """
        Clean up all workspaces (for /cleanall endpoint).
        
        Returns:
            Number of workspaces cleaned up
        """
        cleaned_count = 0
        
        if not self.base_workspace_dir.exists():
            return 0
        
        for workspace_dir in self.base_workspace_dir.iterdir():
            if not workspace_dir.is_dir():
                continue
            
            collection_id = workspace_dir.name
            if self.cleanup_workspace(collection_id):
                cleaned_count += 1
        
        logger.info(f"Cleaned up all {cleaned_count} workspaces")
        return cleaned_count
    
    def list_workspaces(self) -> list[Dict[str, Any]]:
        """
        List all workspaces with their metadata.
        
        Returns:
            List of workspace information dictionaries
        """
        workspaces = []
        
        if not self.base_workspace_dir.exists():
            return workspaces
        
        for workspace_dir in self.base_workspace_dir.iterdir():
            if not workspace_dir.is_dir():
                continue
            
            collection_id = workspace_dir.name
            metadata = self.get_workspace_metadata(collection_id)
            
            workspace_info = {
                "collection_id": collection_id,
                "workspace_path": str(workspace_dir),
                "exists": True,
                "metadata": metadata,
                "is_expired": self.is_workspace_expired(collection_id) if metadata else True
            }
            
            workspaces.append(workspace_info)
        
        return workspaces
    
    def get_workspace_size(self, collection_id: str) -> int:
        """
        Get the size of a workspace in bytes.
        
        Args:
            collection_id: The collection identifier
            
        Returns:
            Size in bytes, or 0 if workspace doesn't exist
        """
        workspace_path = self.get_workspace_path(collection_id)
        
        if not workspace_path.exists():
            return 0
        
        total_size = 0
        try:
            for file_path in workspace_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception as e:
            logger.error(f"Error calculating workspace size for {collection_id}: {e}")
        
        return total_size


# Global workspace manager instance
workspace_manager = WorkspaceManager()

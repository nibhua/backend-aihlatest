from fastapi import APIRouter, HTTPException, Form, Request
from fastapi.responses import FileResponse
from typing import Optional, List, Dict, Any
import os
import json
import uuid
import time
from pathlib import Path
import sys

# Import logging components
from core import api_logger_instance, get_request_id, workspace_manager
from core.concurrency_decorators import podcast_concurrency


from .podcast_generator.main import generate_podcast
from .podcast_generator.models import PodcastRequest

router = APIRouter(prefix="/podcast", tags=["podcast"])

# Paths
VECTOR_DIR = Path("vector_store")  # Legacy fallback
LEGACY_AUDIO_OUTPUT_DIR = Path("podcast_service/audio_output")
LEGACY_AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def _load_chunks(collection_id: str) -> List[Dict[str, Any]]:
    """Load chunks from the collection for content extraction."""
    # Try workspace-based path first
    if collection_id.startswith('col_'):
        try:
            vector_store_dir = workspace_manager.get_vector_store_path(collection_id)
            chunks_path = vector_store_dir / "chunks.json"
            if chunks_path.exists():
                with chunks_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "chunks" in data:
                    data = data["chunks"]
                return data if isinstance(data, list) else []
        except Exception:
            pass  # Fall back to legacy path
    
    # Fall back to legacy path
    chunks_path = VECTOR_DIR / collection_id / "chunks.json"
    if not chunks_path.exists():
        return []
    try:
        with chunks_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "chunks" in data:
            data = data["chunks"]
        return data if isinstance(data, list) else []
    except Exception:
        return []

def _get_audio_output_dir(collection_id: str) -> Path:
    """Get the audio output directory for a collection."""
    if collection_id.startswith('col_'):
        try:
            # Use workspace manager's method
            return workspace_manager.get_audio_output_path(collection_id)
        except Exception:
            pass  # Fall back to legacy path
    
    # Fall back to legacy path
    return LEGACY_AUDIO_OUTPUT_DIR

def _extract_content_from_snippets(snippets: List[Dict[str, Any]]) -> str:
    """Extract and combine content from snippet results for podcast generation."""
    content_parts = []
    
    for snippet in snippets:
        # Extract key information from each snippet
        doc_name = snippet.get("file_name", "Unknown Document")
        heading = snippet.get("heading", "")
        snippet_text = snippet.get("snippet", "")
        
        # Format the content nicely
        if heading:
            content_parts.append(f"From {doc_name}, section '{heading}': {snippet_text}")
        else:
            content_parts.append(f"From {doc_name}: {snippet_text}")
    
    return "\n\n".join(content_parts)

@router.post("/generate_from_snippets")
@podcast_concurrency
async def generate_podcast_from_snippets(
    request: Request,
    snippets: str = Form(...),  # JSON string of snippet results
    podcast_type: str = Form(default="overview"),
    language: str = Form(default="English"),
    selected_text: Optional[str] = Form(default=None),
    theme: Optional[str] = Form(default=None),
    collection_id: Optional[str] = Form(default=None)  # Add collection_id parameter
):
    """
    Generate a podcast from search result snippets.
    """
    start_time = time.time()
    request_id = getattr(request.state, 'request_id', get_request_id())
    
    # Log podcast generation start
    api_logger_instance.log_performance(
        request_id=request_id,
        operation="podcast_snippet_generation_start",
        duration=0,
        podcast_type=podcast_type,
        language=language,
        has_selected_text=bool(selected_text)
    )
    
    try:
        # Parse snippets JSON
        snippet_data = json.loads(snippets)
        if not isinstance(snippet_data, list):
            raise HTTPException(status_code=400, detail="Snippets must be a list")
        
        # Extract content from snippets
        content = _extract_content_from_snippets(snippet_data)
        
        # Add selected text context if provided
        if selected_text:
            content = f"Selected text for context: {selected_text}\n\n{content}"
        
        if not content.strip():
            raise HTTPException(status_code=400, detail="No content found in snippets")
        
        # Generate unique filename
        audio_filename = f"podcast_{uuid.uuid4().hex[:8]}.mp3"
        
        # Use existing podcast generation logic
        audio_file_path, generated_script = await generate_podcast(
            text_input=content,
            podcast_type=podcast_type,
            output_filename=audio_filename,
            language=language,
            theme=theme or "",
            collection_id=collection_id
        )
        
        # Move the generated file to workspace-based audio directory
        source_path = Path(audio_file_path)
        # Use workspace-based path if collection_id is provided, otherwise fall back to legacy
        if collection_id and collection_id.startswith('col_'):
            target_path = _get_audio_output_dir(collection_id) / audio_filename
        else:
            target_path = LEGACY_AUDIO_OUTPUT_DIR / audio_filename
        
        if source_path.exists():
            import shutil
            shutil.move(str(source_path), str(target_path))
            audio_file_path = str(target_path)
        
        processing_time = time.time() - start_time
        
        # Log podcast generation completion
        api_logger_instance.log_performance(
            request_id=request_id,
            operation="podcast_generation_complete",
            duration=processing_time,
            audio_filename=audio_filename,
            snippets_processed=len(snippet_data)
        )
        
        response_data = {
            "message": "Podcast generated successfully from snippets",
            "audio_file_path": audio_file_path,
            "filename": audio_filename,
            "content_summary": f"Generated from {len(snippet_data)} relevant sections",
            "generated_script": generated_script,
            "collection_id": collection_id  # Include collection_id in response
        }

        api_logger_instance.log_info(response_data)
        
        return response_data

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in snippets parameter")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating podcast: {str(e)}")

@router.post("/generate_from_text")
@podcast_concurrency
async def generate_podcast_from_text(
    request: Request,
    text_input: str = Form(...),
    podcast_type: str = Form(default="overview"),
    language: str = Form(default="English"),
    theme: Optional[str] = Form(default=None),
    collection_id: Optional[str] = Form(default=None)  # Add collection_id parameter
):
    """
    Generate a podcast from direct text input.
    """
    start_time = time.time()
    request_id = getattr(request.state, 'request_id', get_request_id())
    
    # Log podcast generation start
    api_logger_instance.log_performance(
        request_id=request_id,
        operation="podcast_text_generation_start",
        duration=0,
        podcast_type=podcast_type,
        language=language,
        text_length=len(text_input)
    )
    
    try:
        if not text_input.strip():
            raise HTTPException(status_code=400, detail="Text input cannot be empty")
        
        # Generate unique filename
        audio_filename = f"podcast_{uuid.uuid4().hex[:8]}.mp3"
        
        # Use existing podcast generation logic
        audio_file_path, generated_script = await generate_podcast(
            text_input=text_input,
            podcast_type=podcast_type,
            output_filename=audio_filename,
            language=language,
            theme=theme or "",
            collection_id=collection_id
        )
        
        # Move the generated file to workspace-based audio directory
        source_path = Path(audio_file_path)
        # Use workspace-based path if collection_id is provided, otherwise fall back to legacy
        if collection_id and collection_id.startswith('col_'):
            target_path = _get_audio_output_dir(collection_id) / audio_filename
        else:
            target_path = LEGACY_AUDIO_OUTPUT_DIR / audio_filename
        
        if source_path.exists():
            import shutil
            shutil.move(str(source_path), str(target_path))
            audio_file_path = str(target_path)
        
        processing_time = time.time() - start_time
        
        # Log podcast generation completion
        api_logger_instance.log_performance(
            request_id=request_id,
            operation="podcast_text_generation_complete",
            duration=processing_time,
            audio_filename=audio_filename
        )
        
        response_data = {
            "message": "Podcast generated successfully",
            "audio_file_path": audio_file_path,
            "filename": audio_filename,
            "generated_script": generated_script,
            "collection_id": collection_id  # Include collection_id in response
        }
        
        return response_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating podcast: {str(e)}")

@router.post("/generate_from_collection")
@podcast_concurrency
async def generate_podcast_from_collection(
    request: Request,
    collection_id: str = Form(...),
    podcast_type: str = Form(default="overview"),
    language: str = Form(default="English"),
    summary_type: str = Form(default="comprehensive"),
    theme: Optional[str] = Form(default=None)
):
    """
    Generate a podcast from an entire collection using collection summary.
    """
    start_time = time.time()
    request_id = getattr(request.state, 'request_id', get_request_id())
    
    # Log podcast generation start
    api_logger_instance.log_performance(
        request_id=request_id,
        operation="podcast_collection_generation_start",
        duration=0,
        collection_id=collection_id,
        podcast_type=podcast_type,
        language=language,
        summary_type=summary_type
    )
    
    try:
        # Validate inputs
        if not collection_id.strip():
            raise HTTPException(status_code=400, detail="collection_id cannot be empty")
        
        # Import collection summarizer
        try:
            from collection_summary_service.summarizer import collection_summarizer
        except ImportError:
            raise HTTPException(status_code=500, detail="Collection summary service not available")
        
        # Generate collection summary (await the async call)
        summary_result = await collection_summarizer.generate_collection_summary(
            collection_id=collection_id,
            summary_type=summary_type
        )
        
        if "error" in summary_result:
            raise HTTPException(status_code=500, detail=summary_result["error"])
        
        # Extract content from summary
        content = ""
        if summary_result.get("summary"):
            if isinstance(summary_result["summary"], str):
                content = summary_result["summary"]
            elif isinstance(summary_result["summary"], dict):
                # If it's a structured summary, extract the main content
                if "summary" in summary_result["summary"]:
                    content = summary_result["summary"]["summary"]
                else:
                    content = json.dumps(summary_result["summary"], ensure_ascii=False)
        
        if not content.strip():
            raise HTTPException(status_code=400, detail="No content found in collection summary")
        
        # Generate unique filename
        audio_filename = f"podcast_collection_{collection_id}_{uuid.uuid4().hex[:8]}.mp3"
        
        # Use existing podcast generation logic
        audio_file_path, generated_script = await generate_podcast(
            text_input=content,
            podcast_type=podcast_type,
            output_filename=audio_filename,
            language=language,
            theme=theme or "",
            collection_id=collection_id
        )
        
        # Move the generated file to workspace-based audio directory
        source_path = Path(audio_file_path)
        target_path = _get_audio_output_dir(collection_id) / audio_filename
        
        # Ensure the target directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Try to move the file from various possible locations
        moved = False
        possible_sources = [
            source_path,
            Path.cwd() / "audio_output" / audio_filename,
            LEGACY_AUDIO_OUTPUT_DIR / audio_filename
        ]
        
        for possible_source in possible_sources:
            if possible_source.exists():
                try:
                    import shutil
                    shutil.move(str(possible_source), str(target_path))
                    audio_file_path = str(target_path)
                    moved = True
                    break
                except Exception as e:
                    # If move fails, try copy and delete
                    try:
                        shutil.copy2(str(possible_source), str(target_path))
                        possible_source.unlink()
                        audio_file_path = str(target_path)
                        moved = True
                        break
                    except Exception:
                        continue
        
        if not moved:
            # If we can't move the file, just use the original path
            pass
        
        processing_time = time.time() - start_time
        
        # Log podcast generation completion
        api_logger_instance.log_performance(
            request_id=request_id,
            operation="podcast_collection_generation_complete",
            duration=processing_time,
            audio_filename=audio_filename,
            collection_id=collection_id,
            summary_type=summary_type
        )
        
        response_data = {
            "message": f"Podcast generated successfully from collection {collection_id}",
            "audio_file_path": audio_file_path,
            "filename": audio_filename,
            "collection_id": collection_id,
            "summary_type": summary_type,
            "content_summary": f"Generated from {summary_result.get('document_count', 0)} documents with {summary_result.get('total_chunks', 0)} chunks",
            "generated_script": generated_script
        }

        api_logger_instance.log_info(response_data)
        return response_data

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating podcast from collection: {str(e)}")

@router.get("/audio/{filename}")
async def get_audio_file(filename: str, collection_id: Optional[str] = None):
    """
    Serve generated audio files.
    """
    # Sanitize filename
    safe_filename = os.path.basename(filename)
    
    # Try to extract collection_id from filename if not provided as query parameter
    if not collection_id and safe_filename.startswith("podcast_collection_"):
        # Extract collection_id from filename pattern: podcast_collection_{collection_id}_{uuid}.mp3
        try:
            # Remove .mp3 extension
            name_without_ext = safe_filename.replace(".mp3", "")
            # Split by underscores
            parts = name_without_ext.split("_")
            if len(parts) >= 4 and parts[0] == "podcast" and parts[1] == "collection":
                # The collection_id is everything between "collection" and the last part (uuid)
                # For filename like "podcast_collection_col_ae61955888_cdcbd62d.mp3"
                # parts = ["podcast", "collection", "col", "ae61955888", "cdcbd62d"]
                # collection_id should be "col_ae61955888" (parts[2] + "_" + parts[3])
                # uuid is "cdcbd62d" (parts[4])
                if len(parts) >= 5:
                    potential_collection_id = "_".join(parts[2:-1])
                    if potential_collection_id.startswith("col_"):
                        collection_id = potential_collection_id
        except Exception:
            pass  # If extraction fails, continue with the original logic
    
    # Try workspace-based path first if collection_id is provided
    if collection_id and collection_id.startswith('col_'):
        try:
            audio_dir = _get_audio_output_dir(collection_id)
            audio_path = audio_dir / safe_filename
            if audio_path.exists():
                return FileResponse(
                    path=str(audio_path),
                    media_type="audio/mpeg",
                    filename=safe_filename
                )
        except Exception:
            pass  # Fall back to legacy path
    
    # Fall back to legacy path
    audio_path = LEGACY_AUDIO_OUTPUT_DIR / safe_filename
    
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        path=str(audio_path),
        media_type="audio/mpeg",
        filename=safe_filename
    )

@router.get("/list_audio")
async def list_audio_files(collection_id: Optional[str] = None):
    """
    List all generated audio files.
    """
    try:
        audio_files = []
        
        # If a specific collection_id is requested, only search in its workspace
        if collection_id and collection_id.startswith('col_'):
            try:
                audio_dir = _get_audio_output_dir(collection_id)
                if audio_dir.exists():
                    for file_path in audio_dir.glob("*.mp3"):
                        stat = file_path.stat()
                        audio_files.append({
                            "filename": file_path.name,
                            "size_bytes": stat.st_size,
                            "created_at": stat.st_ctime,
                            "collection_id": collection_id
                        })
            except Exception:
                pass  # Fall back to legacy path
        else:
            # List audio files from all workspaces and legacy directory
            try:
                # Search in all workspaces
                for workspace_dir in workspace_manager.base_workspace_dir.iterdir():
                    if not workspace_dir.is_dir():
                        continue
                    
                    audio_dir = workspace_dir / "audio_output"
                    if audio_dir.exists():
                        for file_path in audio_dir.glob("*.mp3"):
                            stat = file_path.stat()
                            audio_files.append({
                                "filename": file_path.name,
                                "size_bytes": stat.st_size,
                                "created_at": stat.st_ctime,
                                "collection_id": workspace_dir.name
                            })
            except Exception:
                pass  # Workspace manager not available
            
            # Also search in legacy directory
            if LEGACY_AUDIO_OUTPUT_DIR.exists():
                for file_path in LEGACY_AUDIO_OUTPUT_DIR.glob("*.mp3"):
                    stat = file_path.stat()
                    audio_files.append({
                        "filename": file_path.name,
                        "size_bytes": stat.st_size,
                        "created_at": stat.st_ctime,
                        "collection_id": None  # Legacy files don't have collection_id
                    })
        
        # Sort by created_at descending
        audio_files.sort(key=lambda x: x["created_at"], reverse=True)
        
        return {
            "audio_files": audio_files
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing audio files: {str(e)}")


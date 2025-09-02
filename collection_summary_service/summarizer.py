import os
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv():
        return None
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import google.generativeai as genai
from datetime import datetime

# Import error handling components
from core import (
    handle_gemini_api_call, validate_api_key, check_network_connectivity,
    TimeoutError, RateLimitError, NetworkError, GeminiAPIError, ValidationError
)

# Set up logging
logger = logging.getLogger(__name__)

# Load .env and configure Gemini API only if key is provided
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    try:
        validate_api_key(GEMINI_API_KEY)
        genai.configure(api_key=GEMINI_API_KEY)
    except ValidationError as e:
        print(f"Warning: Invalid API key configuration: {e}")

class CollectionSummarizer:
    """
    Generates comprehensive summaries for entire document collections.
    """
    
    def __init__(self):
    # Allow model override via environment (GEMINI_MODEL or LLM_MODEL)
        self.model = os.getenv("GEMINI_MODEL") or os.getenv("LLM_MODEL") or "gemini-2.0-flash-lite"
        # Legacy summaries directory (for backward compatibility)
        self.legacy_summaries_dir = Path("collection_summary_service/summaries")
        self.legacy_summaries_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()  # Add lock for thread safety
    
    def _get_summaries_dir(self, collection_id: str) -> Path:
        """
        Get the summaries directory for a collection.
        Uses workspace-based path if collection_id starts with 'col_', otherwise uses legacy path.
        """
        if collection_id.startswith('col_'):
            try:
                from core.workspace_manager import workspace_manager
                return workspace_manager.get_summaries_path(collection_id)
            except ImportError:
                # Fallback to legacy path if workspace manager not available
                return self.legacy_summaries_dir
        else:
            return self.legacy_summaries_dir
    
    async def generate_collection_summary(
        self, 
        collection_id: str,
        summary_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of the entire document collection.
        
        Args:
            collection_id: The collection ID to summarize
            summary_type: Type of summary ("comprehensive", "executive", "thematic")
        
        Returns:
            Dictionary containing the collection summary
        """
        async with self._lock:  # Ensure thread safety
            try:
                logger.info(f"Starting summary generation for collection {collection_id}, type {summary_type}")
                
                # Load collection data
                collection_data = self._load_collection_data(collection_id)
                if not collection_data:
                    logger.warning(f"Collection {collection_id} not found or empty")
                    return {"error": "Collection not found or empty"}
                
                # Generate summary based on type
                if summary_type == "comprehensive":
                    summary = await self._generate_comprehensive_summary(collection_data)
                elif summary_type == "executive":
                    summary = await self._generate_executive_summary(collection_data)
                elif summary_type == "thematic":
                    summary = await self._generate_thematic_summary(collection_data)
                else:
                    logger.error(f"Unknown summary type: {summary_type}")
                    return {"error": f"Unknown summary type: {summary_type}"}
                
                # Save summary
                summary_result = {
                    "collection_id": collection_id,
                    "summary_type": summary_type,
                    "summary": summary,
                    "document_count": len(collection_data.get("documents", [])),
                    "total_chunks": len(collection_data.get("chunks", [])),
                    "generated_at": datetime.now().isoformat(),
                    "metadata": {
                        "model": self.model,
                        "documents": [doc.get("name", "Unknown") for doc in collection_data.get("documents", [])]
                }
                }
                
                self._save_summary(collection_id, summary_type, summary_result)
                logger.info(f"Successfully generated {summary_type} summary for collection {collection_id}")
                return summary_result
                
            except Exception as e:
                logger.error(f"Error generating {summary_type} summary for collection {collection_id}: {str(e)}")
                return {"error": f"Error generating collection summary: {str(e)}"}
    
    def _load_collection_data(self, collection_id: str) -> Optional[Dict[str, Any]]:
        """Load collection data from vector store."""
        # Try workspace-based path first if collection_id starts with 'col_'
        if collection_id.startswith('col_'):
            try:
                from core.workspace_manager import workspace_manager
                vector_dir = workspace_manager.get_vector_store_path(collection_id)
            except ImportError:
                # Fallback to legacy path if workspace manager not available
                vector_dir = Path("vector_store") / collection_id
        else:
            vector_dir = Path("vector_store") / collection_id
        
        # Load chunks data
        chunks_path = vector_dir / "chunks.json"
        chunks_data = []
        if chunks_path.exists():
            try:
                with chunks_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "chunks" in data:
                        chunks_data = data["chunks"]
                    elif isinstance(data, list):
                        chunks_data = data
            except Exception:
                pass
        
        # Load metadata
        meta_path = vector_dir / "meta.json"
        meta_data = {}
        if meta_path.exists():
            try:
                with meta_path.open("r", encoding="utf-8") as f:
                    meta_data = json.load(f)
            except Exception:
                pass
        
        if not chunks_data:
            return None
        
        # Group chunks by document
        documents = {}
        for chunk in chunks_data:
            doc_id = chunk.get("doc_id") or chunk.get("source") or chunk.get("file_name", "Unknown")
            if doc_id not in documents:
                documents[doc_id] = {
                    "name": doc_id,
                    "chunks": [],
                    "headings": set(),
                    "content_preview": ""
                }
            
            documents[doc_id]["chunks"].append(chunk)
            
            # Collect headings
            heading = chunk.get("heading_text") or chunk.get("heading", "")
            if heading:
                documents[doc_id]["headings"].add(heading)
            
            # Build content preview
            content = chunk.get("content") or chunk.get("text", "")
            if content and len(documents[doc_id]["content_preview"]) < 1000:
                documents[doc_id]["content_preview"] += content[:200] + " "
        
        # Convert sets to lists for JSON serialization
        for doc in documents.values():
            doc["headings"] = list(doc["headings"])
            doc["content_preview"] = doc["content_preview"].strip()
        
        return {
            "collection_id": collection_id,
            "documents": list(documents.values()),
            "chunks": chunks_data,
            "metadata": meta_data
        }
    
    async def _generate_comprehensive_summary(self, collection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive summary of the collection."""
        
        # Prepare document overviews
        doc_overviews = []
        for doc in collection_data["documents"]:
            overview = f"Document: {doc['name']}\n"
            overview += f"Sections: {', '.join(doc['headings'][:10])}\n"  # Limit headings
            overview += f"Content preview: {doc['content_preview'][:300]}...\n"
            doc_overviews.append(overview)
        
        documents_text = "\n\n".join(doc_overviews)
        
        prompt = f"""
        Analyze this document collection and provide a comprehensive summary.
        
        Collection contains {len(collection_data['documents'])} documents:
        {documents_text}
        
        Provide a comprehensive analysis in this JSON format:
        {{
            "executive_summary": "High-level overview of the entire collection",
            "main_themes": ["Theme 1", "Theme 2", "Theme 3"],
            "key_topics": [
                {{
                    "topic": "Topic name",
                    "description": "What this topic covers",
                    "documents": ["Which documents cover this topic"]
                }}
            ],
            "document_relationships": "How the documents relate to each other",
            "coverage_analysis": "What areas are well-covered vs gaps",
            "insights": [
                {{
                    "insight": "Key insight from the collection",
                    "supporting_documents": ["Documents that support this insight"]
                }}
            ],
            "recommendations": "Recommendations for using this collection"
        }}
        """
        
        return await self._call_gemini(prompt)
    
    async def _generate_executive_summary(self, collection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an executive summary of the collection."""
        
        # Create a condensed view
        doc_summaries = []
        for doc in collection_data["documents"]:
            summary = f"{doc['name']}: {doc['content_preview'][:150]}..."
            doc_summaries.append(summary)
        
        documents_text = "\n".join(doc_summaries)
        
        prompt = f"""
        Create a concise executive summary of this document collection.
        
        Collection overview:
        {documents_text}
        
        Provide an executive summary in this JSON format:
        {{
            "overview": "One paragraph overview of the entire collection",
            "key_points": ["Point 1", "Point 2", "Point 3"],
            "main_focus": "What is the primary focus of this collection",
            "business_value": "What value does this collection provide",
            "quick_insights": ["Insight 1", "Insight 2"]
        }}
        """
        
        return await self._call_gemini(prompt)
    
    async def _generate_thematic_summary(self, collection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a thematic analysis of the collection."""
        
        # Extract all headings and content snippets
        all_headings = []
        content_samples = []
        
        for doc in collection_data["documents"]:
            all_headings.extend(doc["headings"])
            content_samples.append(f"{doc['name']}: {doc['content_preview'][:180]}")
        
        headings_text = ", ".join(all_headings[:40])  # Limit headings
        content_text = "\n".join(content_samples)
        
        prompt = f"""
        Analyze the themes and patterns in this document collection.
        
        Document headings: {headings_text}
        
        Content samples:
        {content_text}
        
        Provide a thematic analysis in this JSON format:
        {{
            "primary_themes": [
                {{
                    "theme": "Theme name",
                    "description": "What this theme encompasses",
                    "prevalence": "How common this theme is",
                    "documents": ["Which documents contain this theme"]
                }}
            ],
            "content_patterns": ["Pattern 1", "Pattern 2"],
            "topic_clusters": [
                {{
                    "cluster": "Cluster name",
                    "topics": ["Related topics in this cluster"]
                }}
            ],
            "cross_document_connections": "How themes connect across documents",
            "thematic_gaps": "What themes or topics are missing"
        }}
        """
        
        return await self._call_gemini(prompt,max_tokens=5000)
    
    async def _call_gemini(self, prompt: str,max_tokens:int=2800) -> Dict[str, Any]:
        """Make a call to Gemini API with proper error handling."""
        try:
            # Check network connectivity first
            if not check_network_connectivity():
                raise NetworkError("No internet connectivity available")
            
            # Create the API call function
            async def gemini_call():
                model = genai.GenerativeModel(self.model)
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.3,
                        max_output_tokens=max_tokens,
                    )
                )
                return response
            
            # Use the error handling wrapper
            response = await handle_gemini_api_call(gemini_call, timeout=150.0)
            
            content = response.text.strip()
            
            # Try to parse JSON response
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {
                    "summary": content,
                    "parse_error": True,
                    "note": "Response was not in JSON format"
                }
                
        except TimeoutError as e:
            return {"error": f"Gemini API timeout: {str(e)}"}
        except RateLimitError as e:
            return {"error": f"Gemini API rate limit exceeded: {str(e)}"}
        except NetworkError as e:
            return {"error": f"Network connectivity issue: {str(e)}"}
        except GeminiAPIError as e:
            return {"error": f"Gemini API error: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}
    
    def _save_summary(self, collection_id: str, summary_type: str, summary_data: Dict[str, Any]):
        """Save summary to disk with atomic write operation."""
        filename = f"{collection_id}_{summary_type}_summary.json"
        summaries_dir = self._get_summaries_dir(collection_id)
        
        # Ensure the summaries directory exists
        summaries_dir.mkdir(parents=True, exist_ok=True)
        
        summary_path = summaries_dir / filename
        temp_path = summaries_dir / f"{filename}.tmp"
        
        try:
            # Write to temporary file first
            with temp_path.open("w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
            # Atomic move to final location
            temp_path.replace(summary_path)
            logger.info(f"Successfully saved {summary_type} summary for collection {collection_id} to {summary_path}")
        except Exception as e:
            # Clean up temp file if it exists
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass
            logger.error(f"Error saving {summary_type} summary for collection {collection_id}: {e}")
            print(f"Error saving summary: {e}")
    
    def get_saved_summary(self, collection_id: str, summary_type: str = "comprehensive") -> Optional[Dict[str, Any]]:
        """Retrieve a previously saved summary."""
        filename = f"{collection_id}_{summary_type}_summary.json"
        summaries_dir = self._get_summaries_dir(collection_id)
        summary_path = summaries_dir / filename
        
        if not summary_path.exists():
            return None
        
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    
    def list_summaries(self, collection_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all available summaries."""
        summaries = []
        
        # If a specific collection_id is requested, only search in its workspace
        if collection_id:
            summaries_dir = self._get_summaries_dir(collection_id)
            if summaries_dir.exists():
                for summary_file in summaries_dir.glob("*.json"):
                    try:
                        with summary_file.open("r", encoding="utf-8") as f:
                            summary_data = json.load(f)
                        
                        summaries.append({
                            "filename": summary_file.name,
                            "collection_id": summary_data.get("collection_id"),
                            "summary_type": summary_data.get("summary_type"),
                            "generated_at": summary_data.get("generated_at"),
                            "document_count": summary_data.get("document_count", 0)
                        })
                    except Exception:
                        continue
        else:
            # List summaries from all workspaces and legacy directory
            try:
                from core.workspace_manager import workspace_manager
                # Search in all workspaces
                for workspace_dir in workspace_manager.base_workspace_dir.iterdir():
                    if not workspace_dir.is_dir():
                        continue
                    
                    summaries_dir = workspace_dir / "summaries"
                    if summaries_dir.exists():
                        for summary_file in summaries_dir.glob("*.json"):
                            try:
                                with summary_file.open("r", encoding="utf-8") as f:
                                    summary_data = json.load(f)
                                
                                summaries.append({
                                    "filename": summary_file.name,
                                    "collection_id": summary_data.get("collection_id"),
                                    "summary_type": summary_data.get("summary_type"),
                                    "generated_at": summary_data.get("generated_at"),
                                    "document_count": summary_data.get("document_count", 0)
                                })
                            except Exception:
                                continue
            except ImportError:
                pass  # Workspace manager not available
            
            # Also search in legacy directory
            if self.legacy_summaries_dir.exists():
                for summary_file in self.legacy_summaries_dir.glob("*.json"):
                    try:
                        with summary_file.open("r", encoding="utf-8") as f:
                            summary_data = json.load(f)
                        
                        summaries.append({
                            "filename": summary_file.name,
                            "collection_id": summary_data.get("collection_id"),
                            "summary_type": summary_data.get("summary_type"),
                            "generated_at": summary_data.get("generated_at"),
                            "document_count": summary_data.get("document_count", 0)
                        })
                    except Exception:
                        continue
        
        # Sort by generated_at descending
        summaries.sort(key=lambda x: x.get("generated_at", ""), reverse=True)
        return summaries

# Global instance
collection_summarizer = CollectionSummarizer()
from fastapi import APIRouter, HTTPException, Form, Request
from typing import Optional, List, Dict, Any
import time
from .summarizer import collection_summarizer

# Import logging components and error handling
from core import (
    api_logger_instance, get_request_id,
    TimeoutError, RateLimitError, NetworkError, GeminiAPIError, ValidationError,
    ResourceNotFoundError
)
from core.concurrency_decorators import summary_concurrency

router = APIRouter(prefix="/collection_summary", tags=["collection_summary"])

@router.post("/generate")
@summary_concurrency
async def generate_collection_summary(
    request: Request,
    collection_id: str = Form(...),
    summary_type: str = Form(default="comprehensive")
):
    """
    Generate a comprehensive summary of the entire document collection.
    
    Args:
        collection_id: The collection ID to summarize
        summary_type: Type of summary ("comprehensive", "executive", "thematic")
    
    Returns:
        Generated collection summary
    """
    start_time = time.time()
    request_id = getattr(request.state, 'request_id', get_request_id())
    
    # Log collection summary generation start
    api_logger_instance.log_performance(
        request_id=request_id,
        operation="collection_summary_generation_start",
        duration=0,
        collection_id=collection_id,
        summary_type=summary_type
    )
    
    try:
        # Validate inputs
        if not collection_id.strip():
            raise HTTPException(status_code=400, detail="collection_id cannot be empty")
        
        valid_types = ["comprehensive", "executive", "thematic"]
        if summary_type not in valid_types:
            raise HTTPException(
                status_code=400, 
                detail=f"summary_type must be one of: {', '.join(valid_types)}"
            )
        
        # Generate summary
        summary_result = await collection_summarizer.generate_collection_summary(
            collection_id=collection_id,
            summary_type=summary_type
        )
        
        # Check for errors
        if "error" in summary_result:
            error_message = summary_result["error"]
            
            # Map error messages to appropriate exceptions
            if "timeout" in error_message.lower():
                raise TimeoutError(error_message)
            elif "rate limit" in error_message.lower():
                raise RateLimitError(error_message)
            elif "network" in error_message.lower() or "connectivity" in error_message.lower():
                raise NetworkError(error_message)
            elif "gemini" in error_message.lower():
                raise GeminiAPIError(error_message)
            else:
                raise HTTPException(status_code=500, detail=error_message)
        
        # Add performance metrics
        processing_time = round((time.time() - start_time) * 1000)  # milliseconds
        summary_result["processing_time_ms"] = processing_time
        
        # Log collection summary generation completion
        api_logger_instance.log_performance(
            request_id=request_id,
            operation="collection_summary_generation_complete",
            duration=processing_time / 1000,
            collection_id=collection_id,
            summary_type=summary_type,
            processing_time_ms=processing_time
        )
        
        return summary_result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating collection summary: {str(e)}")

@router.get("/get/{collection_id}")
async def get_collection_summary(
    collection_id: str,
    summary_type: str = "comprehensive"
):
    """
    Retrieve a previously generated collection summary.
    
    Args:
        collection_id: The collection ID
        summary_type: Type of summary to retrieve
    
    Returns:
        Previously generated summary or 404 if not found
    """
    try:
        summary = collection_summarizer.get_saved_summary(collection_id, summary_type)
        
        if not summary:
            raise HTTPException(
                status_code=404, 
                detail=f"No {summary_type} summary found for collection {collection_id}"
            )
        
        return summary
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving summary: {str(e)}")

@router.get("/list")
async def list_collection_summaries(collection_id: Optional[str] = None):
    """
    List all available collection summaries.
    
    Args:
        collection_id: Optional filter by collection ID
    
    Returns:
        List of available summaries
    """
    try:
        summaries = collection_summarizer.list_summaries(collection_id)
        
        return {
            "summaries": summaries,
            "total_summaries": len(summaries),
            "filtered_by_collection": collection_id is not None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing summaries: {str(e)}")

@router.post("/generate_all_types")
@summary_concurrency
async def generate_all_summary_types(collection_id: str = Form(...)):
    """
    Generate all types of summaries for a collection.
    
    Args:
        collection_id: The collection ID to summarize
    
    Returns:
        Results for all summary types
    """
    start_time = time.time()
    
    try:
        if not collection_id.strip():
            raise HTTPException(status_code=400, detail="collection_id cannot be empty")
        
        summary_types = ["comprehensive", "executive", "thematic"]
        results = {}
        
        for summary_type in summary_types:
            try:
                result = await collection_summarizer.generate_collection_summary(
                    collection_id=collection_id,
                    summary_type=summary_type
                )
                results[summary_type] = result
            except Exception as e:
                results[summary_type] = {"error": str(e)}
        
        # Add performance metrics
        processing_time = round((time.time() - start_time) * 1000)  # milliseconds
        
        return {
            "collection_id": collection_id,
            "results": results,
            "processing_time_ms": processing_time,
            "generated_types": [t for t in summary_types if "error" not in results[t]]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summaries: {str(e)}")

@router.get("/types")
async def get_summary_types():
    """
    Get information about available summary types.
    """
    return {
        "summary_types": [
            {
                "type": "comprehensive",
                "description": "Detailed analysis covering themes, relationships, and insights",
                "use_case": "Deep understanding of the entire collection",
                "typical_length": "Detailed multi-section analysis"
            },
            {
                "type": "executive", 
                "description": "High-level overview focused on key points and business value",
                "use_case": "Quick overview for decision makers",
                "typical_length": "Concise summary with key highlights"
            },
            {
                "type": "thematic",
                "description": "Analysis of themes, patterns, and topic clusters",
                "use_case": "Understanding content organization and thematic structure",
                "typical_length": "Theme-focused analysis with categorization"
            }
        ],
        "recommended_workflow": [
            "1. Generate executive summary for quick overview",
            "2. Generate comprehensive summary for detailed analysis", 
            "3. Generate thematic summary for content organization"
        ]
    }

@router.get("/health")
async def collection_summary_health_check():
    """Health check for collection summary service."""
    return {
        "status": "healthy",
        "model": collection_summarizer.model,
        "available_types": ["comprehensive", "executive", "thematic"]
    }
from fastapi import APIRouter, HTTPException, Form, Request
from typing import List, Dict, Any, Optional
import json
import time
from .generator import insights_generator

# Import logging components
from core import api_logger_instance, get_request_id

router = APIRouter(prefix="/insights", tags=["insights"])

@router.post("/generate")
async def generate_insights(
    request: Request,
    selected_text: str = Form(...),
    relevant_sections: str = Form(...),  # JSON string of relevant sections
    insight_types: Optional[str] = Form(default=None)  # JSON array of insight types
):
    """
    Generate contextual insights from selected text and relevant document sections.
    
    Args:
        selected_text: The text selected by the user
        relevant_sections: JSON string containing relevant sections from documents
        insight_types: Optional JSON array of specific insight types to generate
    
    Returns:
        Generated insights with analysis
    """
    start_time = time.time()
    request_id = getattr(request.state, 'request_id', get_request_id())
    
    # Log insights generation start
    api_logger_instance.log_performance(
        request_id=request_id,
        operation="insights_generation_start",
        duration=0,
        selected_text_length=len(selected_text),
        has_insight_types=bool(insight_types)
    )
    
    try:
        # Parse relevant sections
        try:
            sections_data = json.loads(relevant_sections)
            if not isinstance(sections_data, list):
                raise HTTPException(status_code=400, detail="relevant_sections must be a JSON array")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in relevant_sections")
        
        # Parse insight types if provided
        types_list = None
        if insight_types:
            try:
                types_list = json.loads(insight_types)
                if not isinstance(types_list, list):
                    raise HTTPException(status_code=400, detail="insight_types must be a JSON array")
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON in insight_types")
        
        # Validate inputs
        if not selected_text.strip():
            raise HTTPException(status_code=400, detail="selected_text cannot be empty")
        
        if not sections_data:
            raise HTTPException(status_code=400, detail="No relevant sections provided")
        
        # Generate insights
        insights_result = insights_generator.generate_insights(
            selected_text=selected_text,
            relevant_sections=sections_data,
            insight_types=types_list
        )
        
        # Add performance metrics
        processing_time = round((time.time() - start_time) * 1000)  # milliseconds
        insights_result["processing_time_ms"] = processing_time
        
        # Log insights generation completion
        api_logger_instance.log_performance(
            request_id=request_id,
            operation="insights_generation_complete",
            duration=processing_time / 1000,
            insights_count=len(insights_result.get('insights', [])),
            processing_time_ms=processing_time
        )
        
        return insights_result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating insights: {str(e)}")

@router.post("/generate_quick")
async def generate_quick_insights(
    request: Request,
    selected_text: str = Form(...),
    relevant_sections: str = Form(...),
    max_insights: int = Form(default=3)
):
    """
    Generate a quick subset of insights optimized for speed (under 10 seconds).
    
    Args:
        selected_text: The text selected by the user
        relevant_sections: JSON string containing relevant sections
        max_insights: Maximum number of insight types to generate (default: 3)
    
    Returns:
        Quick insights focused on most important types
    """
    start_time = time.time()
    request_id = getattr(request.state, 'request_id', get_request_id())
    
    # Log quick insights generation start
    api_logger_instance.log_performance(
        request_id=request_id,
        operation="quick_insights_generation_start",
        duration=0,
        selected_text_length=len(selected_text),
        max_insights=max_insights
    )
    
    try:
        # Parse relevant sections
        try:
            sections_data = json.loads(relevant_sections)
            if not isinstance(sections_data, list):
                raise HTTPException(status_code=400, detail="relevant_sections must be a JSON array")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in relevant_sections")
        
        # Validate inputs
        if not selected_text.strip():
            raise HTTPException(status_code=400, detail="selected_text cannot be empty")
        
        # Use only the most important insight types for speed
        quick_types = ["connections", "examples", "contradictions"][:max_insights]
        
        # Generate insights
        insights_result = insights_generator.generate_insights(
            selected_text=selected_text,
            relevant_sections=sections_data,
            insight_types=quick_types
        )
        
        # Add performance metrics
        processing_time = round((time.time() - start_time) * 1000)  # milliseconds
        insights_result["processing_time_ms"] = processing_time
        insights_result["mode"] = "quick"
        
        # Log quick insights generation completion
        api_logger_instance.log_performance(
            request_id=request_id,
            operation="quick_insights_generation_complete",
            duration=processing_time / 1000,
            insights_count=len(insights_result.get('insights', [])),
            processing_time_ms=processing_time
        )
        
        return insights_result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating quick insights: {str(e)}")

@router.get("/types")
async def get_available_insight_types():
    """
    Get list of available insight types that can be generated.
    """
    return {
        "insight_types": [
            {
                "type": "contradictions",
                "description": "Find conflicting viewpoints or contradictory information",
                "example": "Identifies when different sources disagree on a topic"
            },
            {
                "type": "examples", 
                "description": "Find concrete examples, case studies, or illustrations",
                "example": "Locates specific instances or real-world applications"
            },
            {
                "type": "trends",
                "description": "Identify patterns or recurring themes across documents", 
                "example": "Spots common patterns or emerging trends"
            },
            {
                "type": "connections",
                "description": "Find relationships between concepts and ideas",
                "example": "Shows how different concepts relate to each other"
            },
            {
                "type": "implications",
                "description": "Analyze broader significance and consequences",
                "example": "Explains what the information means in a larger context"
            }
        ],
        "recommended_quick_types": ["connections", "examples", "contradictions"],
        "performance_note": "Quick mode uses 3 types for sub-10-second response"
    }


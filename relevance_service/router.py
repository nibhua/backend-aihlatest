from fastapi import APIRouter, HTTPException, Form, Request
from typing import List, Dict, Any, Optional
import json
import time
from .analyzer import relevance_analyzer

# Import logging components
from core import api_logger_instance, get_request_id

router = APIRouter(prefix="/relevance", tags=["relevance"])

@router.post("/analyze_single")
async def analyze_single_relevance(
    request: Request,
    selected_text: str = Form(...),
    relevant_section: str = Form(...),  # JSON string of a single relevant section
    analysis_depth: str = Form(default="standard")  # "quick", "standard", or "detailed"
):
    """
    Analyze why a single document section is relevant to the selected text.
    """
    start_time = time.time()
    request_id = getattr(request.state, 'request_id', get_request_id())

    api_logger_instance.log_performance(
        request_id=request_id,
        operation="single_relevance_analysis_start",
        duration=0,
        analysis_depth=analysis_depth,
        selected_text_length=len(selected_text)
    )

    try:
        # Parse relevant section
        try:
            section_data = json.loads(relevant_section)
            if not isinstance(section_data, dict):
                raise HTTPException(status_code=400, detail="relevant_section must be a JSON object")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in relevant_section")

        # Validate inputs
        if not selected_text.strip():
            raise HTTPException(status_code=400, detail="selected_text cannot be empty")

        if analysis_depth not in ["quick", "standard", "detailed"]:
            raise HTTPException(status_code=400, detail="analysis_depth must be 'quick', 'standard', or 'detailed'")

        # Analyze relevance
        analysis_result = relevance_analyzer.analyze_relevance(
            selected_text=selected_text,
            relevant_section=section_data,
            analysis_depth=analysis_depth
        )

        # Add performance metrics
        processing_time = round((time.time() - start_time) * 1000)
        analysis_result["processing_time_ms"] = processing_time

        api_logger_instance.log_performance(
            request_id=request_id,
            operation="single_relevance_analysis_complete",
            duration=processing_time / 1000,
            processing_time_ms=processing_time
        )

        # IMPORTANT: analysis_result already includes both "relevance_analysis" and "analysis" alias
        return analysis_result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing relevance: {str(e)}")

@router.post("/analyze_multiple")
async def analyze_multiple_relevance(
    request: Request,
    selected_text: str = Form(...),
    relevant_sections: str = Form(...),  # JSON string of relevant sections array
    max_sections: int = Form(default=5)
):
    """
    Analyze relevance for multiple sections and provide comparative insights.
    """
    start_time = time.time()
    request_id = getattr(request.state, 'request_id', get_request_id())

    api_logger_instance.log_performance(
        request_id=request_id,
        operation="multiple_relevance_analysis_start",
        duration=0,
        max_sections=max_sections,
        selected_text_length=len(selected_text)
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

        if not sections_data:
            raise HTTPException(status_code=400, detail="No relevant sections provided")

        if max_sections < 1 or max_sections > 10:
            raise HTTPException(status_code=400, detail="max_sections must be between 1 and 10")

        # Analyze multiple relevance
        analysis_result = relevance_analyzer.analyze_multiple_relevance(
            selected_text=selected_text,
            relevant_sections=sections_data,
            max_sections=max_sections
        )

        # Add performance metrics
        processing_time = round((time.time() - start_time) * 1000)
        analysis_result["processing_time_ms"] = processing_time

        api_logger_instance.log_performance(
            request_id=request_id,
            operation="multiple_relevance_analysis_complete",
            duration=processing_time / 1000,
            sections_analyzed=len(analysis_result.get('individual_analyses', [])),
            processing_time_ms=processing_time
        )

        # Returns "individual_analyses" (each with 'analysis') + "comparative_analysis" (+ 'analysis' alias)
        return analysis_result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing multiple relevance: {str(e)}")

@router.post("/explain_connection")
async def explain_connection(
    request: Request,
    text1: str = Form(...),
    text2: str = Form(...),
    context: Optional[str] = Form(default=None)
):
    """
    Explain the connection between two pieces of text.
    """
    start_time = time.time()
    request_id = getattr(request.state, 'request_id', get_request_id())

    api_logger_instance.log_performance(
        request_id=request_id,
        operation="connection_explanation_start",
        duration=0,
        text1_length=len(text1),
        text2_length=len(text2),
        has_context=bool(context)
    )

    try:
        if not text1.strip() or not text2.strip():
            raise HTTPException(status_code=400, detail="Both text1 and text2 must be provided")

        mock_section = {
            "file_name": "Comparison Text",
            "heading": "Text Comparison",
            "snippet": text2 if not context else f"{text2}\n\nAdditional context: {context}",
            "score": 1.0
        }

        analysis_result = relevance_analyzer.analyze_relevance(
            selected_text=text1,
            relevant_section=mock_section,
            analysis_depth="standard"
        )

        processing_time = round((time.time() - start_time) * 1000)
        analysis_result["processing_time_ms"] = processing_time
        analysis_result["comparison_mode"] = True

        api_logger_instance.log_performance(
            request_id=request_id,
            operation="connection_explanation_complete",
            duration=processing_time / 1000,
            processing_time_ms=processing_time
        )

        return analysis_result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error explaining connection: {str(e)}")

@router.get("/analysis_types")
async def get_analysis_types():
    return {
        "analysis_depths": {
            "quick": {
                "description": "Fast analysis with basic relevance explanation",
                "typical_response_time": "1-2 seconds",
                "use_case": "Real-time feedback during text selection"
            },
            "standard": {
                "description": "Balanced analysis with good detail and reasonable speed",
                "typical_response_time": "2-4 seconds",
                "use_case": "Default analysis for most use cases"
            },
            "detailed": {
                "description": "Comprehensive analysis with extensive insights",
                "typical_response_time": "4-8 seconds",
                "use_case": "Deep research and thorough understanding"
            }
        },
        "relevance_types": [
            "direct_match",
            "conceptual_similarity",
            "contextual_relation",
            "supporting_evidence",
            "contradictory_view"
        ],
        "confidence_levels": ["high", "medium", "low"],
        "max_sections_per_analysis": 10
    }

@router.get("/health")
async def relevance_health_check():
    return {
        "status": "healthy",
        "model": relevance_analyzer.model,
        "available_depths": ["quick", "standard", "detailed"]
    }

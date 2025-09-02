import os
try:
    from dotenv import load_dotenv
except Exception:
    # dotenv is optional in some environments (CI/eval); provide noop
    def load_dotenv():
        return None
import json
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import google.generativeai as genai
import uuid

# Load environment variables from .env (if present) and configure Gemini API
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

class InsightsGenerator:
    """
    Generates contextual insights from selected text and relevant document sections.
    """
    
    def __init__(self):
        self.model = os.getenv("GEMINI_MODEL") or os.getenv("LLM_MODEL") or "gemini-2.0-flash-lite"

    def generate_insights(
        self, 
        selected_text: str, 
        relevant_sections: List[Dict[str, Any]],
        insight_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate insights based on selected text and relevant sections.
        
        Args:
            selected_text: The text selected by the user
            relevant_sections: List of relevant sections from documents
            insight_types: Types of insights to generate (contradictions, examples, trends, etc.)
        
        Returns:
            Dictionary containing generated insights in frontend-compatible format
        """
        if insight_types is None:
            insight_types = ["contradictions", "examples", "trends", "connections", "implications"]
        
        # Prepare context from relevant sections
        context = self._prepare_context(relevant_sections)
        
        # Generate insights for each type
        insights_list = []
        for insight_type in insight_types:
            try:
                insight_data = self._generate_specific_insight(selected_text, context, insight_type)
                if insight_data and not insight_data.get("error"):
                    # Convert to frontend-compatible format
                    frontend_insight = self._convert_to_frontend_format(insight_data, insight_type, relevant_sections)
                    if frontend_insight:
                        insights_list.append(frontend_insight)
            except Exception as e:
                print(f"Error generating {insight_type} insight: {e}")
        
        return {
            "selected_text": selected_text,
            "insights": insights_list,
            "source_sections": len(relevant_sections),
            "timestamp": self._get_timestamp()
        }
    
    def _prepare_context(self, relevant_sections: List[Dict[str, Any]]) -> str:
        """Prepare context string from relevant sections."""
        context_parts = []
        
        for i, section in enumerate(relevant_sections[:10]):  # Limit to top 10 sections
            doc_name = section.get("file_name", f"Document {i+1}")
            heading = section.get("heading", "")
            snippet = section.get("snippet", "")
            
            if heading:
                context_parts.append(f"[{doc_name} - {heading}]\n{snippet}")
            else:
                context_parts.append(f"[{doc_name}]\n{snippet}")
        
        return "\n\n".join(context_parts)
    
    def _generate_specific_insight(self, selected_text: str, context: str, insight_type: str) -> Optional[Dict[str, Any]]:
        """Generate a specific type of insight."""
        
        prompts = {
            "contradictions": f"""
            Analyze the selected text and the provided context to identify any contradictions or conflicting viewpoints.
            
            Selected text: "{selected_text}"
            
            Context from documents:
            {context}
            
            Find contradictions between the selected text and the context, or within the context itself.
            Return your analysis in this JSON format:
            {{
                "found_contradictions": true/false,
                "contradictions": [
                    {{
                        "description": "Brief description of the contradiction",
                        "source1": "First conflicting statement or source",
                        "source2": "Second conflicting statement or source",
                        "significance": "Why this contradiction matters"
                    }}
                ],
                "summary": "Overall summary of contradictory findings"
            }}
            """,
            
            "examples": f"""
            Find concrete examples, case studies, or illustrations related to the selected text from the provided context.
            
            Selected text: "{selected_text}"
            
            Context from documents:
            {context}
            
            Identify relevant examples that support, illustrate, or relate to the selected text.
            Return your analysis in this JSON format:
            {{
                "found_examples": true/false,
                "examples": [
                    {{
                        "description": "Description of the example",
                        "source": "Which document/section this comes from",
                        "relevance": "How this example relates to the selected text",
                        "type": "case_study/illustration/data_point/etc"
                    }}
                ],
                "summary": "Overall summary of examples found"
            }}
            """,
            
            "trends": f"""
            Identify patterns, trends, or recurring themes related to the selected text across the provided context.
            
            Selected text: "{selected_text}"
            
            Context from documents:
            {context}
            
            Look for patterns, trends, or themes that emerge across multiple documents or sections.
            Return your analysis in this JSON format:
            {{
                "found_trends": true/false,
                "trends": [
                    {{
                        "pattern": "Description of the pattern or trend",
                        "frequency": "How often this appears",
                        "sources": ["List of sources where this pattern appears"],
                        "significance": "What this trend indicates or implies"
                    }}
                ],
                "summary": "Overall summary of trends identified"
            }}
            """,
            
            "connections": f"""
            Find connections and relationships between the selected text and other concepts in the provided context.
            
            Selected text: "{selected_text}"
            
            Context from documents:
            {context}
            
            Identify how the selected text connects to other ideas, concepts, or topics in the context.
            Return your analysis in this JSON format:
            {{
                "found_connections": true/false,
                "connections": [
                    {{
                        "concept": "Related concept or idea",
                        "relationship": "How they are connected",
                        "source": "Where this connection is found",
                        "strength": "strong/moderate/weak"
                    }}
                ],
                "summary": "Overall summary of connections found"
            }}
            """,
            
            "implications": f"""
            Analyze the implications, consequences, or broader significance of the selected text based on the provided context.
            
            Selected text: "{selected_text}"
            
            Context from documents:
            {context}
            
            Determine what the selected text implies or what its broader significance might be.
            Return your analysis in this JSON format:
            {{
                "found_implications": true/false,
                "implications": [
                    {{
                        "implication": "What this implies or suggests",
                        "reasoning": "Why this implication follows",
                        "scope": "local/organizational/industry/global",
                        "confidence": "high/medium/low"
                    }}
                ],
                "summary": "Overall summary of implications"
            }}
            """
        }
        
        if insight_type not in prompts:
            return None
        
        try:
            # Prepare the full prompt with system instructions
            full_prompt = f"""You are an expert analyst who generates insights from document content. Always respond with valid JSON as requested.

{prompts[insight_type]}"""
            
            model = genai.GenerativeModel(self.model)
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=3000,
                )
            )
            
            content = response.text.strip()
            
            # Extract JSON from markdown code blocks if present
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
            
            # Try to parse JSON response
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract JSON from the content
                try:
                    # Look for JSON object in the content
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        json_content = content[json_start:json_end]
                        return json.loads(json_content)
                except:
                    pass
                
                # If all parsing fails, return the raw content
                return {"raw_response": content, "parse_error": True}
                
        except Exception as e:
            return {"error": str(e)}
    
    def _convert_to_frontend_format(self, insight_data: Dict[str, Any], insight_type: str, relevant_sections: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Convert backend insight format to frontend-compatible format."""
        
        # Map insight types to frontend types
        type_mapping = {
            "contradictions": "key_findings",
            "examples": "related_concepts", 
            "trends": "trends",
            "connections": "related_concepts",
            "implications": "action_items"
        }
        
        frontend_type = type_mapping.get(insight_type, "key_findings")
        
        # Extract content based on insight type
        content = ""
        findings_count = 0
        has_findings = False
        
        if insight_type == "contradictions":
            has_findings = insight_data.get("found_contradictions", False)
            if has_findings:
                contradictions = insight_data.get("contradictions", [])
                findings_count = len(contradictions)
                if contradictions:
                    content = f"Found {findings_count} contradiction(s): " + insight_data.get("summary", "")
                else:
                    content = insight_data.get("summary", "No contradictions found.")
            else:
                content = insight_data.get("summary", "No contradictions found.")
                
        elif insight_type == "examples":
            has_findings = insight_data.get("found_examples", False)
            if has_findings:
                examples = insight_data.get("examples", [])
                findings_count = len(examples)
                if examples:
                    content = f"Found {findings_count} example(s): " + insight_data.get("summary", "")
                else:
                    content = insight_data.get("summary", "No examples found.")
            else:
                content = insight_data.get("summary", "No examples found.")
                
        elif insight_type == "trends":
            has_findings = insight_data.get("found_trends", False)
            if has_findings:
                trends = insight_data.get("trends", [])
                findings_count = len(trends)
                if trends:
                    content = f"Found {findings_count} trend(s): " + insight_data.get("summary", "")
                else:
                    content = insight_data.get("summary", "No trends found.")
            else:
                content = insight_data.get("summary", "No trends found.")
                
        elif insight_type == "connections":
            has_findings = insight_data.get("found_connections", False)
            if has_findings:
                connections = insight_data.get("connections", [])
                findings_count = len(connections)
                if connections:
                    content = f"Found {findings_count} connection(s): " + insight_data.get("summary", "")
                else:
                    content = insight_data.get("summary", "No connections found.")
            else:
                content = insight_data.get("summary", "No connections found.")
                
        elif insight_type == "implications":
            has_findings = insight_data.get("found_implications", False)
            if has_findings:
                implications = insight_data.get("implications", [])
                findings_count = len(implications)
                if implications:
                    content = f"Found {findings_count} implication(s): " + insight_data.get("summary", "")
                else:
                    content = insight_data.get("summary", "No implications found.")
            else:
                content = insight_data.get("summary", "No implications found.")
        
        if not content:
            return None
        
        # Calculate dynamic confidence based on multiple factors
        confidence = self._calculate_confidence(insight_data, insight_type, findings_count, has_findings, len(relevant_sections))
            
        # Create frontend-compatible insight object
        return {
            "id": str(uuid.uuid4()),
            "type": frontend_type,
            "title": f"{insight_type.title()} Analysis",
            "content": content,
            "confidence": confidence,
            "sources": relevant_sections[:3]  # Use first 3 relevant sections as sources
        }
    
    def _calculate_confidence(self, insight_data: Dict[str, Any], insight_type: str, findings_count: int, has_findings: bool, source_count: int) -> float:
        """Calculate dynamic confidence based on insight quality and content."""
        
        # Base confidence starts at 0.5
        confidence = 0.5
        
        # Factor 1: Whether findings were found (major impact)
        if has_findings:
            confidence += 0.2
        else:
            # If no findings, reduce confidence but don't make it too low
            confidence -= 0.1
        
        # Factor 2: Number of findings (more findings = higher confidence)
        if findings_count > 0:
            # Cap the bonus at 0.15 for findings count
            findings_bonus = min(findings_count * 0.03, 0.15)
            confidence += findings_bonus
        
        # Factor 3: Source count (more sources = higher confidence)
        if source_count > 0:
            # Cap the bonus at 0.1 for source count
            source_bonus = min(source_count * 0.02, 0.1)
            confidence += source_bonus
        
        # Factor 4: Content quality (longer, more detailed content = higher confidence)
        summary = insight_data.get("summary", "")
        if summary:
            # Longer summaries tend to be more detailed and confident
            summary_length = len(summary)
            if summary_length > 100:
                confidence += 0.05
            elif summary_length > 50:
                confidence += 0.03
        
        # Factor 5: Insight type specific adjustments
        if insight_type == "contradictions":
            # Contradictions are harder to find, so finding them is more significant
            if has_findings and findings_count > 0:
                confidence += 0.05
        elif insight_type == "trends":
            # Trends require pattern recognition, so finding them is valuable
            if has_findings and findings_count > 0:
                confidence += 0.03
        elif insight_type == "implications":
            # Implications are speculative, so moderate confidence
            confidence -= 0.05
        
        # Ensure confidence is within bounds [0.1, 0.95]
        confidence = max(0.1, min(0.95, confidence))
        
        # Round to 2 decimal places
        return round(confidence, 2)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()

# Global instance
insights_generator = InsightsGenerator()


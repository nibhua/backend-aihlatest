import os
import json
from typing import List, Dict, Any
import google.generativeai as genai
from datetime import datetime
import re

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv():
        return None

# Load .env and configure Gemini API only if key is present
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


class RelevanceAnalyzer:
    """
    Analyzes and explains why document sections are relevant to selected text.
    """

    def __init__(self):
        # Use 2.5 Flash by default (can be overridden via env)
        self.model = os.getenv("GEMINI_MODEL") or os.getenv("LLM_MODEL") or "gemini-2.5-flash"

    # ------------------------------- utils ---------------------------------

    def _coerce_json(self, content: str) -> Dict[str, Any]:
        """
        Turn possibly fenced/double-encoded JSON into a dict.
        Handles:
        - ```json ... ```
        - surrounding noise
        - double-encoded JSON strings
        """
        try:
            if isinstance(content, dict):
                return content

            s = (content or "").strip()

            # Strip markdown code fences
            s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
            s = re.sub(r"\s*```$", "", s)

            # Strip leading 'json'
            s = re.sub(r'^\s*"?json"?\s*', "", s, flags=re.IGNORECASE)

            # Slice to the outermost {...}
            l = s.find("{")
            r = s.rfind("}")
            if l != -1 and r != -1 and r > l:
                s = s[l:r + 1]

            # Clean trailing commas to be defensive
            s = re.sub(r",\s*}", "}", s)
            s = re.sub(r",\s*]", "]", s)

            # Try up to two decode passes (handles double-encoded strings)
            for _ in range(2):
                if isinstance(s, str):
                    try:
                        s = json.loads(s)
                        continue
                    except Exception:
                        break

            if isinstance(s, dict):
                return s

            return {
                "relevance_type": "analysis_error",
                "explanation": str(content),
                "confidence": "low",
                "parse_error": True,
            }
        except Exception:
            return {
                "relevance_type": "analysis_error",
                "explanation": str(content),
                "confidence": "low",
                "parse_error": True,
            }

    def _gen_config(self, max_tokens: int, temperature: float = 0.3):
        """
        Unified generation config that asks for raw JSON.
        """
        return genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
        )

    # --------------------------- public methods ----------------------------

    def analyze_relevance(
        self,
        selected_text: str,
        relevant_section: Dict[str, Any],
        analysis_depth: str = "standard",
    ) -> Dict[str, Any]:
        """
        Analyze why a specific section is relevant to the selected text.
        """
        try:
            analysis = self._generate_relevance_explanation(
                selected_text,
                relevant_section,
                analysis_depth,
            )

            # normalize to dict
            analysis = analysis if isinstance(analysis, dict) else self._coerce_json(str(analysis))

            result = {
                "selected_text": selected_text,
                "relevant_section": {
                    "file_name": relevant_section.get("file_name", "Unknown"),
                    "heading": relevant_section.get("heading", ""),
                    "snippet": relevant_section.get("snippet", ""),
                    "page": relevant_section.get("page"),
                    "score": relevant_section.get("score"),
                },
                "relevance_analysis": analysis,
                "analysis_depth": analysis_depth,
                "timestamp": datetime.now().isoformat(),
            }

            # Alias for FE compatibility
            result["analysis"] = result["relevance_analysis"]
            return result

        except Exception as e:
            err = {
                "relevance_type": "error",
                "explanation": f"Error analyzing relevance: {str(e)}",
                "confidence": "low",
                "error": True,
            }
            return {
                "selected_text": selected_text,
                "relevant_section": relevant_section,
                "relevance_analysis": err,
                "analysis": err,  # alias
                "analysis_depth": analysis_depth,
                "timestamp": datetime.now().isoformat(),
            }

    def analyze_multiple_relevance(
        self,
        selected_text: str,
        relevant_sections: List[Dict[str, Any]],
        max_sections: int = 5,
    ) -> Dict[str, Any]:
        """
        Analyze relevance for multiple sections and provide comparative insights.
        """
        sections_to_analyze = relevant_sections[:max_sections]
        # Each entry already contains "analysis" alias because it calls analyze_relevance()
        individual_analyses = [self.analyze_relevance(selected_text, s, "quick") for s in sections_to_analyze]

        comparative_analysis = self._generate_comparative_analysis(
            selected_text, sections_to_analyze
        )

        if not isinstance(comparative_analysis, dict):
            comparative_analysis = self._coerce_json(str(comparative_analysis))

        result = {
            "selected_text": selected_text,
            "individual_analyses": individual_analyses,
            "comparative_analysis": comparative_analysis,
            "sections_analyzed": len(sections_to_analyze),
            "timestamp": datetime.now().isoformat(),
        }
        # Optional alias for consumers that look for "analysis" on multi endpoint
        result["analysis"] = result["comparative_analysis"]
        return result

    # --------------------------- LLM prompts -------------------------------

    def _generate_relevance_explanation(
        self,
        selected_text: str,
        section: Dict[str, Any],
        depth: str,
    ) -> Dict[str, Any]:
        """Generate explanation for why a section is relevant."""
        doc_name = section.get("file_name", "Unknown Document")
        heading = section.get("heading", "")
        snippet = section.get("snippet", "")

        ctx = f"Document: {doc_name}\n"
        if heading:
            ctx += f"Section: {heading}\n"
        ctx += f"Content: {snippet}"

        if depth == "quick":
            schema_text = """
{
  "relevance_type": "direct_match|conceptual_similarity|contextual_relation|supporting_evidence",
  "explanation": "1–2 sentence explanation of why this is relevant",
  "key_connections": ["connection1", "connection2"],
  "supporting_evidence": [],
  "confidence": "high|medium|low"
}
"""
            max_tokens = 700
        elif depth == "detailed":
            schema_text = """
{
  "relevance_type": "direct_match|conceptual_similarity|contextual_relation|supporting_evidence|contradictory_view",
  "explanation": "Detailed explanation of the relevance",
  "key_connections": ["specific connections"],
  "shared_concepts": ["concepts present in both texts"],
  "relationship_strength": "strong|moderate|weak",
  "supporting_evidence": [],
  "confidence": "high|medium|low",
  "additional_insights": "extra insights",
  "potential_follow_up": "suggested follow-ups"
}
"""
            max_tokens = 2000
        else:  # standard
            schema_text = """
{
  "relevance_type": "direct_match|conceptual_similarity|contextual_relation|supporting_evidence|contradictory_view",
  "explanation": "Clear explanation of why this section is relevant",
  "key_connections": ["specific connections between the texts"],
  "shared_concepts": ["concepts that appear in both"],
  "supporting_evidence": [],
  "confidence": "high|medium|low",
  "relevance_score_interpretation": "What the similarity score suggests"
}
"""
            max_tokens = 1200

        prompt = f"""
You are an expert at analyzing document relevance.

Return ONLY valid JSON matching the schema.
Do not include any commentary or markdown fences.
The first character of your output must be '{{' and the last must be '}}'.

Selected text: "{selected_text}"

Document section:
{ctx}

Schema:
{schema_text}
"""

        try:
            model = genai.GenerativeModel(self.model)
            response = model.generate_content(
                prompt,
                generation_config=self._gen_config(max_tokens=max_tokens, temperature=0.3),
            )
            return self._coerce_json(response.text)
        except Exception as e:
            return {
                "relevance_type": "error",
                "explanation": str(e),
                "confidence": "low",
                "error": True,
            }

    def _generate_comparative_analysis(
        self,
        selected_text: str,
        sections: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate comparative analysis across multiple sections."""
        summaries = []
        for i, section in enumerate(sections):
            name = section.get("file_name", f"Document {i+1}")
            heading = section.get("heading", "")
            snippet = (section.get("snippet", "") or "").strip()
            score = section.get("score", 0)
            snippet_short = snippet[:200] + ("..." if len(snippet) > 200 else "")
            title = f"{name}" + (f" - {heading}" if heading else "")
            summaries.append(f"{i+1}. {title} (Score: {score:.3f})\n   {snippet_short}")

        sections_text = "\n\n".join(summaries)

        schema_text = """
{
  "overall_theme": "Common theme connecting the sections to the selected text",
  "most_relevant": { "section_number": 1, "reason": "Why this section is most relevant" },
  "relevance_patterns": ["Pattern 1", "Pattern 2"],
  "coverage_analysis": "What aspects are covered",
  "gaps": "What is not well covered",
  "recommendation": "Which sections to prioritize and why"
}
"""

        prompt = f"""
You are an expert at comparative document analysis.

Return ONLY valid JSON matching the schema.
Do not include any commentary or markdown fences.

Selected text: "{selected_text}"

Document sections:
{sections_text}

Schema:
{schema_text}
"""

        try:
            model = genai.GenerativeModel(self.model)
            response = model.generate_content(
                prompt,
                generation_config=self._gen_config(max_tokens=1400, temperature=0.3),
            )
            return self._coerce_json(response.text)
        except Exception as e:
            return {
                "overall_theme": "Error in comparative analysis",
                "error": str(e),
            }


# Global instance
relevance_analyzer = RelevanceAnalyzer()

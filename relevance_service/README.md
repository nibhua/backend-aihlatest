# relevance_service/

Analyzes and explains the relevance between selected text and document sections.

## What it does

- **Relevance Analysis**: Explains why document sections are relevant to selected text
- **Single/Multiple Analysis**: Analyzes relevance for individual or multiple sections
- **Depth Control**: Provides quick, standard, or detailed analysis levels
- **Comparative Insights**: Compares relevance across multiple sections

## Files

- `analyzer.py` — Core relevance analysis logic and AI model integration
- `router.py` — FastAPI endpoints for relevance analysis operations
- `__init__.py` — Module exports

## Analysis Types

- **Single Section**: Detailed analysis of why one section is relevant
- **Multiple Sections**: Comparative analysis across multiple relevant sections
- **Quick Analysis**: Fast, high-level relevance assessment
- **Detailed Analysis**: Comprehensive relevance explanation with examples

## Analysis Depth

- **Quick**: Fast, high-level relevance assessment
- **Standard**: Balanced analysis with key relevance factors
- **Detailed**: Comprehensive analysis with examples and context

## Endpoints

- `POST /relevance/analyze_single` — Analyze relevance of single section
- `POST /relevance/analyze_multiple` — Compare relevance across multiple sections

# collection_summary_service/

Generates comprehensive summaries of document collections using AI.

## What it does

- **Collection Summarization**: Creates high-level summaries of entire document collections
- **Multiple Summary Types**: Generates comprehensive, executive, and thematic summaries
- **AI-Powered Analysis**: Uses language models to extract key themes and insights
- **Caching**: Stores generated summaries for efficient retrieval

## Files

- `summarizer.py` — Core summarization logic and AI model integration
- `router.py` — FastAPI endpoints for summary operations
- `summaries/` — Directory containing cached summary files
- `__init__.py` — Module exports

## Summary Types

- **Comprehensive**: Detailed analysis covering all major topics
- **Executive**: High-level overview for decision makers
- **Thematic**: Focused on key themes and patterns

## Endpoints

- `POST /collection_summary/generate` — Generate new collection summary
- `GET /collection_summary/get/{collection_id}` — Retrieve existing summary

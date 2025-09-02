# insights_service/

Generates contextual insights and analysis from selected text and document sections.

## What it does

- **Text Analysis**: Analyzes selected text in the context of relevant document sections
- **Insight Generation**: Creates various types of insights (key points, connections, implications)
- **Contextual Understanding**: Leverages document context for deeper analysis
- **Quick Insights**: Provides fast, focused insights for immediate use

## Files

- `generator.py` — Core insight generation logic and AI model integration
- `router.py` — FastAPI endpoints for insight operations
- `__init__.py` — Module exports

## Insight Types

- **Key Points**: Main takeaways and important information
- **Connections**: Relationships between selected text and document content
- **Implications**: Potential consequences and broader significance
- **Questions**: Thought-provoking questions for deeper exploration

## Endpoints

- `POST /insights/generate` — Generate comprehensive insights
- `POST /insights/generate_quick` — Generate focused, quick insights
- `GET /insights/history` — Retrieve insight generation history

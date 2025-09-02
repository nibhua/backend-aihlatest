# snippets/

Generates text snippets for search results and document previews.

## What it does

- **Snippet Generation**: Creates compact, informative text snippets from document chunks
- **Smart Selection**: Chooses the most relevant sentences based on scoring algorithms
- **Length Control**: Ensures snippets fit within specified character limits
- **Boilerplate Filtering**: Removes common section headers when appropriate

## Files

- `generator.py` — Core snippet generation logic and scoring algorithms

## Key Features

- **Sentence Scoring**: Ranks sentences based on relevance and length
- **Term Preference**: Prioritizes sentences containing specific terms
- **IDF Weighting**: Uses inverse document frequency for better term selection
- **Length Optimization**: Prefers mid-length sentences for readability

## Scoring Factors

- **Term Presence**: Bonus for sentences containing preferred terms
- **IDF Weight**: Higher scores for rare, important terms
- **Length**: Optimal length range (60-240 characters)
- **Content Quality**: Avoids boilerplate headings when possible

## Usage

```python
from snippets.generator import make_snippet

# Generate snippet for a chunk
snippet = make_snippet(
    chunk=chunk_data,
    prefer_terms=["important", "key"],
    max_chars=200
)
```

# persona/

Handles persona-based search and constraint extraction for personalized document retrieval.

## What it does

- **Intent Detection**: Analyzes user queries to detect search intent and preferences
- **Constraint Extraction**: Identifies dietary, travel, and other constraints from queries
- **Search Optimization**: Adjusts search parameters based on detected persona and constraints
- **Query Enhancement**: Improves search results through persona-aware processing

## Files

- `router.py` — Core persona detection and constraint extraction logic

## Key Features

- **Task Rules**: Pattern-based intent detection for different query types
- **Constraint Recognition**: Identifies dietary restrictions, travel preferences, etc.
- **Search Tuning**: Adjusts k-value, MMR, and other search parameters
- **Fallback Support**: Built-in fallback constraints for common scenarios

## Supported Constraints

- **Dietary**: Vegetarian, vegan, gluten-free, etc.
- **Travel Planning**: Itinerary, accommodation, transport preferences
- **Analysis Types**: Comparison, study, overview, detailed analysis

## Usage

The persona module automatically processes search queries to optimize results based on detected intent and constraints.

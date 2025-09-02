# Podcast Generator API

A modular podcast generation system using Gemini 2.0 Flash-Lite API and Azure TTS, with two types of podcasts (overview and conversational), integrated into a FastAPI application.

## Features

- **Overview Podcast**: Generates a 2-4 minute summary podcast from selected sections or entire knowledgebase in 10 different languages. 
- **Conversational Podcast**: Creates a conversation between two AI speakers (Ava and Andrew) about the input text
- **FastAPI Integration**: RESTful API endpoints for podcast generation
- **Azure TTS Integration**: Uses Azure Cognitive Services Speech SDK for high-quality text-to-speech
- **Gemini AI Integration**: Uses Google's Gemini 2.0 Flash-Lite API for content generation

## Project Structure

```
podcast_project/
├── main.py                     # FastAPI application entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── test_api.py                 # API testing script
├── audio_output/               # Generated podcast files
└── podcast_generator/          # Core podcast generation module
    ├── __init__.py
    ├── main.py                 # Core podcast generation logic
    ├── models.py               # Pydantic models
    ├── tts_utils.py            # Azure TTS utilities
    └── gemini_utils.py         # Gemini API utilities
```

## Installation

1. Install dependencies:
```bash
uv venv
uv pip install -r requirements.txt
```

2. Set environment variables (optional, defaults are provided):
```bash
export AZURE_SPEECH_KEY="your_azure_speech_key"
export AZURE_SPEECH_REGION="your_azure_region"
export GEMINI_API_KEY="your_gemini_api_key"
```

## Usage

### Starting the Server

```bash
python main.py
```

The server will start on `http://0.0.0.0:8000`

### API Endpoints

#### GET /
Returns a welcome message.

#### POST /generate_podcast
Generates a podcast from input text.

**Request Body:**
```json
{
    "text_input": "Your text content here",
    "podcast_type": "overview",  // or "conversational"
    "output_filename": "my_podcast.mp3"
}
```

**Response:**
```json
{
    "message": "Podcast generated successfully",
    "audio_file_path": "audio_output/my_podcast.mp3"
}
```

### Example Usage

#### Overview Podcast
```bash
curl -X POST "http://localhost:8000/generate_podcast" \
  -H "Content-Type: application/json" \
  -d '{
    "text_input": "Artificial Intelligence is transforming the world. Machine learning algorithms are being used in various industries.",
    "podcast_type": "overview",
    "output_filename": "ai_overview.mp3"
  }'
```

#### Conversational Podcast
```bash
curl -X POST "http://localhost:8000/generate_podcast" \
  -H "Content-Type: application/json" \
  -d '{
    "text_input": "Climate change is one of the most pressing issues of our time.",
    "podcast_type": "conversational",
    "output_filename": "climate_discussion.mp3"
  }'
```

## Testing

Run the test script to verify the API is working:

```bash
python test_api.py
```

## Configuration

The system uses the following default configurations:

- **Azure Speech Key**: Provided in the code (can be overridden with environment variable)
- **Azure Speech Region**: centralindia
- **Gemini API Key**: Provided in the code (can be overridden with environment variable)
- **Voices**: 
  - Ava Multilingual Neural (en-US-AvaMultilingualNeural) for overview and Speaker A
  - Andrew Multilingual Neural (en-US-AndrewMultilingualNeural) for Speaker B

## Audio Output

Generated podcasts are saved in the `audio_output/` directory as MP3 files. The system automatically handles:

- Text chunking for long content
- Audio file combination for conversational podcasts
- Temporary file cleanup

## Dependencies

- fastapi: Web framework
- uvicorn: ASGI server
- pydantic: Data validation
- requests: HTTP client
- pydub: Audio processing
- google-generativeai: Gemini API client
- azure-cognitiveservices-speech: Azure Speech SDK

## Notes

- The system requires internet connectivity for both Gemini API and Azure Speech Services
- Audio files are generated in MP3 format
- The conversational podcast feature parses speaker dialogue and assigns different voices
- Long text inputs are automatically chunked and combined for optimal audio generation


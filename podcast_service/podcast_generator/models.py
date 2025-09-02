from pydantic import BaseModel

class PodcastRequest(BaseModel):
    text_input: str
    podcast_type: str # 'overview' or 'conversational'
    output_filename: str
    language: str = "en-US" # Default to English

class PodcastResponse(BaseModel):
    message: str
    audio_file_path: str

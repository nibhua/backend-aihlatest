import os
import asyncio
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional
import hashlib
import json
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv():
        return None
from .tts_utils import generate_audio_azure_speech, generate_audio_chunked, get_voice_name
from .gemini_utils import configure_gemini, generate_text

# Load environment variables from .env (if present)
load_dotenv()

# Environment variables for Azure Speech and Gemini API (no hardcoded defaults)
AZURE_TTS_KEY = os.getenv("AZURE_TTS_KEY")
AZURE_TTS_ENDPOINT = os.getenv("AZURE_TTS_ENDPOINT")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL") or os.getenv("LLM_MODEL")

# Import performance configuration
from .config import (
    MAX_CONCURRENT_PODCASTS,
    CACHE_ENABLED,
    CACHE_EXPIRY_HOURS
)

# Configure Gemini API only if key is provided
if GEMINI_API_KEY:
    configure_gemini(GEMINI_API_KEY)

# Global concurrency control - use a larger thread pool to avoid deadlocks
_podcast_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PODCASTS)
_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_PODCASTS * 2)  # Double the workers to prevent deadlocks

def _generate_cache_key(text_input: str, podcast_type: str, language: str, theme: str = "") -> str:
    """Generate a cache key for podcast content."""
    content = f"{text_input}:{podcast_type}:{language}:{theme}"
    return hashlib.md5(content.encode()).hexdigest()

def _get_cached_podcast(cache_key: str, collection_id: str = None) -> Optional[tuple[str, str]]:
    """Get cached podcast if available and not expired."""
    if not CACHE_ENABLED:
        return None
    
    # Use collection-specific cache directory if collection_id is provided
    if collection_id and collection_id.startswith('col_'):
        from core.workspace_manager import workspace_manager
        cache_dir = workspace_manager.get_workspace_path(collection_id) / "podcast_cache"
    else:
        cache_dir = Path("podcast_cache")
    
    cache_file = cache_dir / f"{cache_key}.json"
    if not cache_file.exists():
        return None
    
    try:
        with cache_file.open("r") as f:
            cache_data = json.load(f)
        
        # Check if cache is expired
        if time.time() - cache_data["timestamp"] > (CACHE_EXPIRY_HOURS * 3600):
            cache_file.unlink()  # Remove expired cache
            return None
        
        audio_file = cache_data["audio_file"]
        script = cache_data["script"]
        
        # Verify audio file still exists
        if Path(audio_file).exists():
            print(f"Using cached podcast: {cache_key}")
            return audio_file, script
        else:
            cache_file.unlink()  # Remove invalid cache
            return None
            
    except Exception:
        # If cache is corrupted, remove it
        try:
            cache_file.unlink()
        except Exception:
            pass
        return None

def _cache_podcast(cache_key: str, audio_file: str, script: str, collection_id: str = None):
    """Cache podcast for future use."""
    if not CACHE_ENABLED:
        return
    
    try:
        # Use collection-specific cache directory if collection_id is provided
        if collection_id and collection_id.startswith('col_'):
            from core.workspace_manager import workspace_manager
            cache_dir = workspace_manager.get_workspace_path(collection_id) / "podcast_cache"
        else:
            cache_dir = Path("podcast_cache")
        
        # Ensure cache directory exists
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        cache_file = cache_dir / f"{cache_key}.json"
        cache_data = {
            "timestamp": time.time(),
            "audio_file": audio_file,
            "script": script
        }
        with cache_file.open("w") as f:
            json.dump(cache_data, f)
        print(f"Cached podcast: {cache_key}")
    except Exception as e:
        print(f"Failed to cache podcast: {e}")

async def generate_overview_podcast(text_input: str, output_filename: str, language: str, theme: str = "", collection_id: str = None) -> tuple[str, str]:
    """Generate overview podcast with caching and async processing."""
    cache_key = _generate_cache_key(text_input, "overview", language, theme)
    
    # Check cache first
    cached_result = _get_cached_podcast(cache_key, collection_id)
    if cached_result:
        return cached_result
    
    # Run heavy processing in thread pool to avoid blocking
    # Note: Router-level concurrency control is already in place
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                _executor,
                _generate_overview_podcast_sync,
                text_input, output_filename, language, theme, collection_id
            ),
            timeout=300.0  # 5 minute timeout
        )
        
        # Cache the result
        _cache_podcast(cache_key, result[0], result[1], collection_id)
        return result
    except asyncio.TimeoutError:
        raise RuntimeError("Podcast generation timed out after 5 minutes")

def _generate_overview_podcast_sync(text_input: str, output_filename: str, language: str, theme: str = "", collection_id: str = None) -> tuple[str, str]:
    """Synchronous version of overview podcast generation."""
    # Use Gemini to create a highly natural overview podcast script
    theme_prompt = ""
    if theme:
        theme_prompt = f"\n\nTHEME REQUIREMENT: Make this a {theme}."
    
    prompt = f"""Create a highly natural and engaging {language} podcast overview script for the following text. Make it sound like a real male human podcaster speaking naturally with these characteristics:

    SPEAKING STYLE:
    - Use natural speech patterns with appropriate filler words and expressions common in {language}
    - Include conversational tone as if speaking directly to a friend
    - Add natural pauses indicated by "..." for emphasis or thinking
    - Use contractions and informal language naturally as spoken in {language}
    - Include occasional verbal reactions like "you know," "actually," or equivalent in {language}
    - Use varied sentence lengths - some short for impact, some longer for explanation
    - Add natural transitions between topics using phrases appropriate to {language}

    CONTENT GUIDELINES:
    - DO NOT include any stage directions or production notes
    - STRICTLY WRITE THE ENTIRE SCRIPT IN {language} ONLY - DO NOT MIX LANGUAGES OR USE OTHER LANGUAGE WORDS
    - Use only native {language} vocabulary, expressions, and sentence structures
    - Avoid any English words, phrases, or mixed language patterns
    - Ensure all technical terms are properly translated to {language}
    - Write as a native {language} speaker would naturally speak
    - Make each speaker have distinct personalities and perspectives
    - Include natural topic transitions using phrases appropriate to {language}
    - Add genuine curiosity and follow-up questions
    - Include personal anecdotes or relatable examples when appropriate
    - Make the conversation feel spontaneous, not scripted
    - Ensure the total length is suitable for a 3-5 minute podcast
    - Stay informative while being conversational and engaging
    - Use cultural references and expressions that are natural for {language} speakers
    - DO NOT INCLUDE ANY READING INSTRUCTIONS, STAGE DIRECTIONS, OR ACTION DESCRIPTIONS LIKE *READS*, *CHUCKLES*, *PAUSES*, *LAUGHS*, ETC.

    Text to discuss without any stage directions or action descriptions:
    {text_input}"""
    script = generate_text(prompt)

    # Get voice name based on language and role
    voice_name = get_voice_name(language, "overview")

    # Use collection-specific audio output path if collection_id is provided
    if collection_id and collection_id.startswith('col_'):
        from core.workspace_manager import workspace_manager
        audio_output_dir = workspace_manager.get_audio_output_path(collection_id)
        audio_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = audio_output_dir / output_filename
    else:
        output_path = Path("audio_output") / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate audio using Azure Speech TTS
    audio_file = generate_audio_chunked(
        script,
        str(output_path),
        voice_name=voice_name,
        speech_key=AZURE_TTS_KEY,
        target_uri=AZURE_TTS_ENDPOINT
    )
    return audio_file, script

async def generate_conversational_podcast(text_input: str, output_filename: str, language: str, collection_id: str = None) -> tuple[str, str]:
    """Generate conversational podcast with caching and async processing."""
    cache_key = _generate_cache_key(text_input, "conversational", language)
    
    # Check cache first
    cached_result = _get_cached_podcast(cache_key, collection_id)
    if cached_result:
        return cached_result
    
    # Run heavy processing in thread pool to avoid blocking
    # Note: Router-level concurrency control is already in place
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                _executor,
                _generate_conversational_podcast_sync,
                text_input, output_filename, language, collection_id
            ),
            timeout=300.0  # 5 minute timeout
        )
        
        # Cache the result
        _cache_podcast(cache_key, result[0], result[1], collection_id)
        return result
    except asyncio.TimeoutError:
        raise RuntimeError("Podcast generation timed out after 5 minutes")

def _generate_conversational_podcast_sync(text_input: str, output_filename: str, language: str, collection_id: str = None) -> tuple[str, str]:
    """Synchronous version of conversational podcast generation."""
    # Use Gemini to create a conversational script between two AI speakers
    prompt = f"""Create a highly natural and engaging conversational script between two speakers in {language} discussing the following text. Make it sound like a real human conversation with these characteristics:

    CONVERSATION STYLE:
    - Use natural speech patterns with appropriate filler words and expressions common in {language}
    - Include natural interruptions and overlapping thoughts
    - Add conversational reactions and acknowledgments appropriate to {language} culture
    - Use contractions and informal language naturally as spoken in {language}
    - Include natural pauses indicated by "..." for dramatic effect or thinking
    - Add occasional laughter and reactions with appropriate filler words and expressions common in {language}
    - Use varied sentence lengths - some short, some long, just like real speech

    FORMAT REQUIREMENTS:
    Speaker A: [First speaker's dialogue with natural speech patterns]
    Speaker B: [Second speaker's dialogue with natural reactions and filler words]
    Speaker A: [First speaker's response with natural flow]
    Speaker B: [Second speaker's response continuing the natural conversation]

    CONTENT GUIDELINES:
    - DO NOT include any stage directions or production notes
    - STRICTLY WRITE THE ENTIRE SCRIPT IN {language} ONLY - DO NOT MIX LANGUAGES OR USE OTHER LANGUAGE WORDS
    - Use only native {language} vocabulary, expressions, and sentence structures
    - Avoid any English words, phrases, or mixed language patterns
    - Ensure all technical terms are properly translated to {language}
    - Write as a native {language} speaker would naturally speak
    - Make each speaker have distinct personalities and perspectives
    - Include natural topic transitions using phrases appropriate to {language}
    - Add genuine curiosity and follow-up questions
    - Include personal anecdotes or relatable examples when appropriate
    - Make the conversation feel spontaneous, not scripted
    - Ensure the total length is suitable for a 3-5 minute podcast
    - Stay informative while being conversational and engaging
    - Use cultural references and expressions that are natural for {language} speakers
    - DO NOT INCLUDE ANY READING INSTRUCTIONS, STAGE DIRECTIONS, OR ACTION DESCRIPTIONS LIKE *READS*, *CHUCKLES*, *PAUSES*, *LAUGHS*, ETC.

    Text to discuss without any stage directions or action descriptions:
    {text_input}"""
    
    script = generate_text(prompt)
    
    # Use collection-specific audio output path if collection_id is provided
    if collection_id and collection_id.startswith('col_'):
        from core.workspace_manager import workspace_manager
        audio_output_dir = workspace_manager.get_audio_output_path(collection_id)
        audio_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = audio_output_dir / output_filename
    else:
        output_path = Path("audio_output") / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    combined_audio = None

    lines = script.split("\n")
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("Speaker A:"):
            speaker_text = line.replace("Speaker A:", "").strip()
            voice = get_voice_name(language, "speaker_a")
        elif line.startswith("Speaker B:"):
            speaker_text = line.replace("Speaker B:", "").strip()
            voice = get_voice_name(language, "speaker_b")
        else:
            continue # Skip lines that don't start with Speaker A or Speaker B

        if speaker_text:
            temp_audio_file = output_path.parent / f"temp_speaker_{i}.wav"
            generate_audio_azure_speech(
                speaker_text,
                str(temp_audio_file),
                voice_name=voice,
                speech_key=AZURE_TTS_KEY,
                target_uri=AZURE_TTS_ENDPOINT
            )
            from pydub import AudioSegment
            segment = AudioSegment.from_file(temp_audio_file, format="wav")
            if combined_audio is None:
                combined_audio = segment
            else:
                combined_audio += segment
            os.remove(temp_audio_file) # Clean up temporary file

    if combined_audio:
        combined_audio.export(str(output_path), format="mp3")
        print(f"Conversational podcast saved to: {output_path}")
        return str(output_path), script
    else:
        raise RuntimeError("No audio generated for conversational podcast.")


async def generate_podcast(text_input: str, podcast_type: str, output_filename: str, language: str, theme: str = "", collection_id: str = None) -> tuple[str, str]:
    """Main podcast generation function with async support and caching."""
    if podcast_type == "overview":
        return await generate_overview_podcast(text_input, output_filename, language, theme, collection_id)
    elif podcast_type == "conversational":
        return await generate_conversational_podcast(text_input, output_filename, language, collection_id)
    elif podcast_type == "news":
        return await generate_news_podcast(text_input, output_filename, language, theme, collection_id)
    elif podcast_type == "story":
        return await generate_story_podcast(text_input, output_filename, language, theme, collection_id)
    else:
        raise ValueError("Invalid podcast_type. Must be 'overview', 'conversational', 'news', or 'story'.")

# Add additional podcast types for variety
async def generate_news_podcast(text_input: str, output_filename: str, language: str, theme: str = "") -> tuple[str, str]:
    """Generate news-style podcast with caching and async processing."""
    cache_key = _generate_cache_key(text_input, "news", language, theme)
    
    # Check cache first
    cached_result = _get_cached_podcast(cache_key)
    if cached_result:
        return cached_result
    
    # Run heavy processing in thread pool to avoid blocking
    # Note: Router-level concurrency control is already in place
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                _executor,
                _generate_news_podcast_sync,
                text_input, output_filename, language, theme
            ),
            timeout=300.0  # 5 minute timeout
        )
        
        # Cache the result
        _cache_podcast(cache_key, result[0], result[1])
        return result
    except asyncio.TimeoutError:
        raise RuntimeError("Podcast generation timed out after 5 minutes")

def _generate_news_podcast_sync(text_input: str, output_filename: str, language: str, theme: str = "") -> tuple[str, str]:
    """Synchronous version of news podcast generation."""
    theme_prompt = ""
    if theme:
        theme_prompt = f"\n\nTHEME REQUIREMENT: Make this a {theme}."
    
    prompt = f"""Create a professional news-style podcast script in {language} for the following content. Make it sound like a professional news anchor with these characteristics:

    STYLE:
    - Professional and authoritative tone
    - Clear pronunciation and pacing
    - Use news-style language and transitions
    - Include relevant context and background information
    - Structure with headline, body, and conclusion
    - Use formal but accessible language

    CONTENT GUIDELINES:
    - DO NOT include any stage directions or production notes
    - STRICTLY WRITE THE ENTIRE SCRIPT IN {language} ONLY
    - Use only native {language} vocabulary and expressions
    - Ensure all technical terms are properly translated to {language}
    - Make it suitable for a 3-5 minute news segment
    - Include natural transitions between topics

    Text to report on:
    {text_input}{theme_prompt}"""
    
    script = generate_text(prompt)
    
    # Get voice name based on language and role
    voice_name = get_voice_name(language, "overview")

    # Generate audio using Azure Speech TTS
    output_path = Path("audio_output") / output_filename
    audio_file = generate_audio_chunked(
        script,
        str(output_path),
        voice_name=voice_name,
        speech_key=AZURE_TTS_KEY,
        target_uri=AZURE_TTS_ENDPOINT
    )
    return audio_file, script

async def generate_story_podcast(text_input: str, output_filename: str, language: str, theme: str = "") -> tuple[str, str]:
    """Generate story-style podcast with caching and async processing."""
    cache_key = _generate_cache_key(text_input, "story", language, theme)
    
    # Check cache first
    cached_result = _get_cached_podcast(cache_key)
    if cached_result:
        return cached_result
    
    # Run heavy processing in thread pool to avoid blocking
    # Note: Router-level concurrency control is already in place
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                _executor,
                _generate_story_podcast_sync,
                text_input, output_filename, language, theme
            ),
            timeout=300.0  # 5 minute timeout
        )
        
        # Cache the result
        _cache_podcast(cache_key, result[0], result[1])
        return result
    except asyncio.TimeoutError:
        raise RuntimeError("Podcast generation timed out after 5 minutes")

def _generate_story_podcast_sync(text_input: str, output_filename: str, language: str, theme: str = "") -> tuple[str, str]:
    """Synchronous version of story podcast generation."""
    theme_prompt = ""
    if theme:
        theme_prompt = f"\n\nTHEME REQUIREMENT: Make this a {theme}."
    
    prompt = f"""Create a compelling story-telling podcast script in {language} for the following content. Make it sound like a master storyteller with these characteristics:

    STYLE:
    - Engaging and immersive narrative
    - Use storytelling techniques like hooks, suspense, and resolution
    - Include vivid descriptions and emotional elements
    - Use varied pacing and rhythm
    - Create a sense of journey and discovery

    CONTENT GUIDELINES:
    - DO NOT include any stage directions or production notes
    - STRICTLY WRITE THE ENTIRE SCRIPT IN {language} ONLY
    - Use only native {language} vocabulary and expressions
    - Ensure all technical terms are properly translated to {language}
    - Make it suitable for a 3-5 minute story
    - Include natural narrative flow and transitions

    Content to tell as a story:
    {text_input}{theme_prompt}"""
    
    script = generate_text(prompt)
    
    # Get voice name based on language and role
    voice_name = get_voice_name(language, "overview")

    # Generate audio using Azure Speech TTS
    output_path = Path("audio_output") / output_filename
    audio_file = generate_audio_chunked(
        script,
        str(output_path),
        voice_name=voice_name,
        speech_key=AZURE_TTS_KEY,
        target_uri=AZURE_TTS_ENDPOINT
    )
    return audio_file, script
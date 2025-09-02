"""
Configuration file for podcast generation performance settings.
"""

import os
from pathlib import Path

# Performance optimization settings
MAX_CONCURRENT_PODCASTS = int(os.getenv("MAX_CONCURRENT_PODCASTS", "3"))
CACHE_ENABLED = os.getenv("PODCAST_CACHE_ENABLED", "true").lower() == "true"
CACHE_DIR = Path("podcast_cache")
CACHE_EXPIRY_HOURS = int(os.getenv("PODCAST_CACHE_EXPIRY_HOURS", "24"))

# Audio processing settings
MAX_AUDIO_CHUNK_SIZE = int(os.getenv("MAX_AUDIO_CHUNK_SIZE", "3000"))
MAX_AUDIO_WORKERS = int(os.getenv("MAX_AUDIO_WORKERS", "4"))

# TTS settings
TTS_TIMEOUT_SECONDS = int(os.getenv("TTS_TIMEOUT_SECONDS", "30"))
TTS_RETRY_ATTEMPTS = int(os.getenv("TTS_RETRY_ATTEMPTS", "3"))

# Gemini API settings
GEMINI_TIMEOUT_SECONDS = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "60"))
GEMINI_MAX_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", "4000"))

# Ensure cache directory exists
if CACHE_ENABLED:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Performance monitoring
ENABLE_PERFORMANCE_MONITORING = os.getenv("ENABLE_PODCAST_PERFORMANCE_MONITORING", "true").lower() == "true"
PERFORMANCE_LOG_INTERVAL = int(os.getenv("PODCAST_PERFORMANCE_LOG_INTERVAL", "10"))  # seconds

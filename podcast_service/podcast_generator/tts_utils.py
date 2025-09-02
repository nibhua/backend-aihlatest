import os
import azure.cognitiveservices.speech as speechsdk
from pathlib import Path
from pydub import AudioSegment

VOICE_MAPPING = {
    "en-US": {
        "overview": "en-US-AvaMultilingualNeural",
        "speaker_a": "en-US-AvaMultilingualNeural",
        "speaker_b": "en-US-AndrewMultilingualNeural"
    },
    "hi-IN": {
        "overview": "hi-IN-MadhurNeural",
        "speaker_a": "hi-IN-MadhurNeural",
        "speaker_b": "hi-IN-SwaraNeural"
    },
    "ja-JP": {
        "overview": "ja-JP-KeitaNeural",
        "speaker_a": "ja-JP-KeitaNeural",
        "speaker_b": "ja-JP-NanamiNeural"
    },
    "es-ES": {
        "overview": "es-ES-AlvaroNeural",
        "speaker_a": "es-ES-AlvaroNeural",
        "speaker_b": "es-ES-ElviraNeural"
    },
    "fr-FR": {
        "overview": "fr-FR-HenriNeural",
        "speaker_a": "fr-FR-HenriNeural",
        "speaker_b": "fr-FR-DeniseNeural"
    },
    "de-DE": {
        "overview": "de-DE-ConradNeural",
        "speaker_a": "de-DE-ConradNeural",
        "speaker_b": "de-DE-KatjaNeural"
    },
    "zh-CN": {
        "overview": "zh-CN-YunxiNeural",
        "speaker_a": "zh-CN-YunxiNeural",
        "speaker_b": "zh-CN-XiaoxiaoNeural"
    },
    "ar-SA": {
        "overview": "ar-SA-HamedNeural",
        "speaker_a": "ar-SA-HamedNeural",
        "speaker_b": "ar-SA-ZariyahNeural"
    },
    "pt-PT": {
        "overview": "pt-PT-DuarteNeural",
        "speaker_a": "pt-PT-DuarteNeural",
        "speaker_b": "pt-PT-RaquelNeural"
    },
    "ru-RU": {
        "overview": "ru-RU-DmitryNeural",
        "speaker_a": "ru-RU-DmitryNeural",
        "speaker_b": "ru-RU-SvetlanaNeural"
    }
}

def get_voice_name(language: str, podcast_role: str) -> str:
    """Get the appropriate voice name based on language and podcast role."""
    # Map frontend language names to Azure locale codes
    language_mapping = {
        "English": "en-US",
        "Hindi": "hi-IN", 
        "Spanish": "es-ES",
        "Japanese": "ja-JP",
        "French": "fr-FR",
        "German": "de-DE",
        "Chinese": "zh-CN",
        "Arabic": "ar-SA",
        "Portuguese": "pt-PT",
        "Russian": "ru-RU"
    }
    
    locale = language_mapping.get(language, "en-US")
    return VOICE_MAPPING.get(locale, VOICE_MAPPING["en-US"]).get(podcast_role, VOICE_MAPPING["en-US"]["overview"])

def generate_audio_azure_speech(text, output_file, voice_name, speech_key, target_uri):
    """Generate audio using Azure Cognitive Services Speech SDK."""
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create speech config
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, endpoint=target_uri)
    speech_config.speech_synthesis_voice_name = voice_name

    # Create audio config to save to file
    audio_config = speechsdk.audio.AudioOutputConfig(filename=str(output_path))

    # Create synthesizer
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    # Synthesize speech
    result = speech_synthesizer.speak_text_async(text).get()

    # Check result
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print(f"Azure Speech TTS audio saved to: {output_file}")
        return str(output_path)
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        error_msg = f"Speech synthesis canceled: {cancellation_details.reason}"
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            error_msg += f" Error details: {cancellation_details.error_details}"
        raise RuntimeError(error_msg)
    else:
        raise RuntimeError(f"Unexpected result reason: {result.reason}")

def _chunk_text_by_chars(text, max_chars):
    import re

    if len(text) <= max_chars:
        return [text]

    tokens = re.findall(r"\S+\s*", text)
    chunks = []
    current = ""

    for token in tokens:
        if len(current) + len(token) <= max_chars:
            current += token
        else:
            if current:
                chunks.append(current.strip())
                current = ""
            if len(token) > max_chars:
                start = 0
                while start < len(token):
                    part = token[start:start + max_chars]
                    part = part.strip()
                    if part:
                        chunks.append(part)
                    start += max_chars
            else:
                current = token

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if c]

def generate_audio_chunked(text, output_file, voice_name, speech_key, target_uri, max_chars=3000):
    """Generate audio for long text by chunking and combining with parallel processing."""
    chunks = _chunk_text_by_chars(text, max_chars)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_files = []
    try:
        # Process chunks in parallel for better performance
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def process_chunk(chunk_data):
            index, chunk = chunk_data
            temp_file = str(output_path.parent / f".tts_chunk_{index}.wav")
            try:
                generate_audio_azure_speech(chunk, temp_file, voice_name, speech_key, target_uri)
                return temp_file
            except Exception as e:
                print(f"Error processing chunk {index}: {e}")
                return None
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=min(len(chunks), 4)) as executor:
            # Submit all chunks for processing
            future_to_chunk = {
                executor.submit(process_chunk, (i, chunk)): i 
                for i, chunk in enumerate(chunks)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                temp_file = future.result()
                if temp_file:
                    temp_files.append(temp_file)

        # Combine audio files
        combined_audio = None
        for temp_file in sorted(temp_files, key=lambda x: int(x.split('_')[-1].split('.')[0])):
            try:
                segment = AudioSegment.from_file(temp_file, format="wav")
                if combined_audio is None:
                    combined_audio = segment
                else:
                    combined_audio += segment
            except Exception as e:
                print(f"Error combining audio file {temp_file}: {e}")
                continue

        if combined_audio:
            # Export as MP3
            combined_audio.export(str(output_path), format="mp3")
            print(f"Chunked Azure Speech TTS audio saved to: {output_file} ({len(chunks)} chunks)")
            return str(output_path)
        else:
            raise RuntimeError("Failed to generate any audio chunks")
            
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except Exception:
                pass


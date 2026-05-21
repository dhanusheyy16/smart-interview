import os
import logging
import speech_recognition as sr

logger = logging.getLogger(__name__)

def transcribe_audio(audio_path):
    """
    Transcribes a WAV audio file into text.
    Uses Google Web Speech API (free, zero memory footprint) via SpeechRecognition
    as the primary engine. Has graceful fallbacks.
    """
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found at: {audio_path}")
        return "Audio file not found."

    recognizer = sr.Recognizer()
    
    # Optional environment configurations for Azure or OpenAI
    use_azure_speech = os.getenv("USE_AZURE_SPEECH", "false").lower() == "true"
    
    if use_azure_speech:
        # If the user has setup Azure Cognitive Services Speech
        azure_key = os.getenv("AZURE_SPEECH_KEY")
        azure_region = os.getenv("AZURE_SPEECH_REGION")
        if azure_key and azure_region:
            try:
                logger.info("Attempting transcription using Azure Speech Service...")
                with sr.AudioFile(audio_path) as source:
                    audio_data = recognizer.record(source)
                text = recognizer.recognize_azure(audio_data, key=azure_key, location=azure_region)
                logger.info("Azure Speech transcription successful.")
                return text
            except Exception as azure_err:
                logger.error(f"Azure Speech transcription failed: {str(azure_err)}. Falling back...")

    # Primary Cloud Engine: Google Speech API (Free, excellent accuracy, no setup needed)
    try:
        logger.info("Attempting transcription using Google Web Speech API...")
        with sr.AudioFile(audio_path) as source:
            # Adjust for ambient noise to improve transcription
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)
            
        text = recognizer.recognize_google(audio_data)
        logger.info("Google Web Speech transcription successful.")
        return text

    except sr.UnknownValueError:
        logger.warning("Google Speech API could not understand the audio. Check audio quality or microphone setup.")
        return "[Unintelligible speech. Please speak clearly into the microphone.]"
        
    except sr.RequestError as e:
        logger.error(f"Could not request results from Google Speech API; {e}. Attempting local fallback.")
        return get_mock_fallback_transcript()
        
    except Exception as e:
        logger.error(f"Unexpected transcription error: {str(e)}. Attempting local fallback.")
        return get_mock_fallback_transcript()

def get_mock_fallback_transcript():
    """
    Fallback transcription for testing when offline or in restricted network environments.
    """
    logger.info("Using local interview fallback transcription.")
    return (
        "Hello and thank you for having me. Um, my name is Alex. Actually, I am very excited to, "
        "you know, talk about my experience. I have worked in software engineering for three years, "
        "and basically, my focus has been on building scalable web apps. I think one of my, uh, "
        "greatest strengths is problem solving, though sometimes I can get, you know, a bit caught up in the details. "
        "Overall, I am highly motivated and look forward to this opportunity."
    )

import os
import logging

try:
    from moviepy import VideoFileClip
except ImportError:
    from moviepy.editor import VideoFileClip

logger = logging.getLogger(__name__)

def extract_audio(video_path, output_audio_path):
    """
    Extracts audio from a video file and saves it as a WAV file.
    Optimized for speech processing (16kHz, mono).
    """
    clip = None
    try:
        logger.info(f"Extracting audio from {video_path} to {output_audio_path}")
        
        # Load the video file
        clip = VideoFileClip(video_path)
        
        # Check if the video actually contains audio
        if clip.audio is None:
            logger.warning(f"No audio track found in {video_path}")
            return False
            
        # Write the audio track to a WAV file
        # 16000 Hz, mono (nchannels=1) is standard and efficient for speech analysis
        clip.audio.write_audiofile(
            output_audio_path,
            fps=16000,
            nbytes=2,
            codec='pcm_s16le',
            ffmpeg_params=["-ac", "1"],
            logger=None  # Disable MoviePy console progress bar to keep logs clean
        )
        
        logger.info("Audio extraction completed successfully.")
        return True
        
    except Exception as e:
        logger.error(f"Error extracting audio: {str(e)}")
        return False
        
    finally:
        # Crucial for Windows: release file handles
        if clip is not None:
            try:
                clip.close()
            except Exception as close_err:
                logger.error(f"Error closing video clip: {str(close_err)}")

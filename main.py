import os
import uuid
import json
import logging
from flask import Flask, render_template, request, jsonify, redirect, url_for
from dotenv import load_dotenv

# Import analysis modules
from modules.audio_extraction import extract_audio
from modules.speech_to_text import transcribe_audio
from modules.audio_analysis import analyze_audio
from modules.video_analysis import analyze_video
from modules.text_analysis import analyze_text_local, get_ai_evaluation

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "smart-interview-super-secret-key-1337")

# Define directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESULTS_FOLDER = os.path.join(BASE_DIR, "results")

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULTS_FOLDER"] = RESULTS_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64MB max upload limit

def cleanup_files(*filepaths):
    """Safely deletes temporary video and audio files to conserve disk space on Render."""
    for filepath in filepaths:
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.info(f"Cleaned up temp file: {filepath}")
            except Exception as e:
                logger.error(f"Failed to delete temp file {filepath}: {str(e)}")

@app.route("/", methods=["GET"])
def index():
    """Renders the main landing page for uploading or recording videos."""
    has_api_key = bool(os.getenv("AZURE_OPENAI_API_KEY"))
    return render_template("index.html", has_api_key=has_api_key)

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Accepts video file, orchestrates processing pipeline, 
    saves result JSON, and returns redirect parameters.
    """
    if "video" not in request.files:
        logger.error("No video file uploaded in request.")
        return jsonify({"success": False, "error": "No video file provided."}), 400

    video_file = request.files["video"]
    if video_file.filename == "":
        logger.error("Uploaded file has empty filename.")
        return jsonify({"success": False, "error": "Empty filename."}), 400

    # Generate unique ID for this analysis session
    session_id = str(uuid.uuid4())
    
    # Save the video locally
    # Supports both mp4 and webm uploads (webcam uses webm commonly)
    extension = os.path.splitext(video_file.filename)[1]
    if not extension:
        extension = ".webm" # Default for stream blobs
    
    video_filename = f"{session_id}{extension}"
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], video_filename)
    audio_filename = f"{session_id}.wav"
    audio_path = os.path.join(app.config["UPLOAD_FOLDER"], audio_filename)
    
    try:
        logger.info(f"Saving uploaded video to: {video_path}")
        video_file.save(video_path)
        
        # 1. Extract Audio
        logger.info("Executing Pipeline Step 1/5: Extracting audio...")
        audio_extracted = extract_audio(video_path, audio_path)
        
        transcript = ""
        audio_metrics = {}
        if audio_extracted:
            # 2. Speech-to-Text
            logger.info("Executing Pipeline Step 2/5: Transcribing audio...")
            transcript = transcribe_audio(audio_path)
            
            # 3. Audio Analysis
            logger.info("Executing Pipeline Step 3/5: Running audio acoustic analysis...")
            audio_metrics = analyze_audio(audio_path, transcript)
        else:
            logger.warning("Audio extraction failed. Running default audio fallback values.")
            transcript = "[No audio track found in the uploaded video file.]"
            audio_metrics = {
                "duration": 5.0,
                "wpm": 0,
                "mean_pitch": 0.0,
                "pitch_variability": 0.0,
                "pause_count": 0,
                "silence_percentage": 100.0,
                "filler_words_count": 0,
                "filler_words_details": {}
            }

        # 4. Video Analysis (OpenCV face & posture checks)
        logger.info("Executing Pipeline Step 4/5: Running OpenCV face/gesture analysis...")
        video_metrics = analyze_video(video_path)
        
        # 5. Local Text Analysis & Azure OpenAI evaluation
        logger.info("Executing Pipeline Step 5/5: Running linguistic and AI assessment...")
        text_metrics = analyze_text_local(transcript)
        
        # Compile aggregate report via Azure OpenAI (or custom fallback)
        final_report = get_ai_evaluation(transcript, audio_metrics, video_metrics, text_metrics)
        
        # Construct complete aggregated results payload
        aggregated_results = {
            "session_id": session_id,
            "transcript": transcript,
            "audio_metrics": audio_metrics,
            "video_metrics": video_metrics,
            "text_metrics": text_metrics,
            "report": final_report
        }
        
        # Write findings to JSON to serve results page
        result_path = os.path.join(app.config["RESULTS_FOLDER"], f"{session_id}.json")
        with open(result_path, "w") as rf:
            json.dump(aggregated_results, rf, indent=4)
            
        logger.info(f"Session {session_id} successfully compiled.")
        
        # Success! Clean up heavy media files immediately to free space
        cleanup_files(video_path, audio_path)
        
        return jsonify({
            "success": True, 
            "redirect_url": url_for("result", session_id=session_id)
        })

    except Exception as e:
        logger.error(f"Pipeline crashed for session {session_id}: {str(e)}")
        # Make sure files are cleaned up even on crash
        cleanup_files(video_path, audio_path)
        return jsonify({"success": False, "error": f"Internal server analysis error: {str(e)}"}), 500


@app.route("/result/<session_id>", methods=["GET"])
def result(session_id):
    """Renders the premium interactive result dashboard for a given session."""
    result_path = os.path.join(app.config["RESULTS_FOLDER"], f"{session_id}.json")
    
    if not os.path.exists(result_path):
        logger.error(f"Requested session result not found: {session_id}")
        return render_template("index.html", error="The requested interview analysis could not be found.")

    try:
        with open(result_path, "r") as rf:
            data = json.load(rf)
        
        return render_template(
            "result.html", 
            session_id=session_id,
            transcript=data["transcript"],
            audio=data["audio_metrics"],
            video=data["video_metrics"],
            text=data["text_metrics"],
            report=data["report"]
        )
    except Exception as e:
        logger.error(f"Error loading result session {session_id}: {str(e)}")
        return render_template("index.html", error="An error occurred while loading your report details.")


if __name__ == "__main__":
    # Run server locally on port 5000
    # On Render, it will run via Gunicorn
    app.run(host="0.0.0.0", port=5000, debug=True)

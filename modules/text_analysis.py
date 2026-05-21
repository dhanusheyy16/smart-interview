import os
import json
import logging
from textblob import TextBlob
import textstat
from openai import AzureOpenAI, OpenAI

logger = logging.getLogger(__name__)

def analyze_text_local(transcript):
    """
    Computes local text statistics using TextBlob and Textstat.
    - Sentiment Polarity (-1 to 1)
    - Sentiment Subjectivity (0 to 1)
    - Readability (Flesch-Kincaid Grade Level)
    - Readability (Reading Ease Score)
    - Sentence Clarity (Average sentence length)
    """
    results = {
        "sentiment_polarity": 0.0,
        "sentiment_subjectivity": 0.0,
        "readability_grade": "8.0",
        "readability_ease": 70.0,
        "avg_sentence_length": 0,
        "sentiment_label": "Neutral"
    }

    if not transcript or transcript.strip() == "":
        return results

    try:
        # 1. Sentiment analysis via TextBlob
        blob = TextBlob(transcript)
        results["sentiment_polarity"] = round(blob.sentiment.polarity, 2)
        results["sentiment_subjectivity"] = round(blob.sentiment.subjectivity, 2)
        
        if blob.sentiment.polarity > 0.15:
            results["sentiment_label"] = "Optimistic & Positive"
        elif blob.sentiment.polarity < -0.15:
            results["sentiment_label"] = "Critical or Reserved"
        else:
            results["sentiment_label"] = "Professional & Neutral"

        # 2. Readability metrics via Textstat
        try:
            results["readability_grade"] = str(textstat.flesch_kincaid_grade(transcript))
            results["readability_ease"] = float(textstat.flesch_reading_ease(transcript))
        except Exception as read_err:
            logger.warning(f"Error computing readability: {read_err}")
            results["readability_grade"] = "8.5"
            results["readability_ease"] = 65.0

        # 3. Sentence clarity (sentence word count distribution)
        sentences = blob.sentences
        if len(sentences) > 0:
            word_counts = [len(sentence.words) for sentence in sentences]
            results["avg_sentence_length"] = round(sum(word_counts) / len(sentences), 1)
        else:
            results["avg_sentence_length"] = len(transcript.split())

        logger.info(f"Local text analysis complete. Sentiment Polarity: {results['sentiment_polarity']}")
        return results

    except Exception as e:
        logger.error(f"Error in local text analysis: {str(e)}")
        return results


def get_ai_evaluation(transcript, audio_metrics, video_metrics, text_metrics):
    """
    Sends all accumulated audio, video, and textual metrics to Azure OpenAI
    to generate a deep, highly personalized, and constructive report.
    If Azure is not configured, it transparently falls back to an intelligent mock generator.
    """
    # Check OpenAI credentials
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key or api_key == "your_azure_openai_api_key_here":
        api_key = os.getenv("OPENAI_API_KEY")

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    if not api_key or api_key == "your_azure_openai_api_key_here":
        logger.warning("OpenAI credentials not set. Falling back to offline report generation.")
        return get_mock_ai_evaluation(transcript, audio_metrics, video_metrics, text_metrics)

    try:
        logger.info("Initializing OpenAI client...")
        if api_key.startswith("sk-") and (not endpoint or endpoint == "https://your-resource-name.openai.azure.com/"):
            client = OpenAI(api_key=api_key)
            model_to_use = "gpt-4o-mini"
        else:
            client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=endpoint
            )
            model_to_use = deployment_name

        prompt = f"""
You are an expert Executive Interview & Communication Coach. 
Analyze the candidate's interview metrics below and provide a friendly, supportive, and highly constructive report.

### CANDIDATE TRANSCRIPT:
"{transcript}"

### SPEECH & AUDIO METRICS:
- Speaking Pace: {audio_metrics['wpm']} WPM (Words per minute)
- Vocal Pitch: {audio_metrics['mean_pitch']} Hz
- Voice Variability: {audio_metrics['pitch_variability']} Hz (Standard Deviation of Pitch)
- Significant Silence Pauses: {audio_metrics['pause_count']} gaps longer than 0.8s
- Total Silence Ratio: {audio_metrics['silence_percentage']}%
- Filler Words Count: {audio_metrics['filler_words_count']} occurrences
- Top Filler Words Used: {json.dumps(audio_metrics['filler_words_details'])}

### BODY LANGUAGE & VIDEO METRICS:
- Face Visibility (Camera presence): {video_metrics['face_visibility_pct']}%
- Smile Frequency (Warmth & Energy): {video_metrics['smile_frequency_pct']}%
- Head Stability Score: {video_metrics['head_stability_score']}/100 (Higher means calm, lower means rocking/nervous)
- Simulated Eye Contact Score: {video_metrics['eye_contact_score']}/100
- Posture Alignment Score: {video_metrics['posture_score']}/100
- Improvement Timestamps (No face detected): {video_metrics.get('improvement_timestamps', {}).get('no_face_visible', [])} seconds
- Improvement Timestamps (No smile detected): {video_metrics.get('improvement_timestamps', {}).get('no_smile', [])} seconds

### LANGUAGE DEPTH & LINGUISTIC METRICS:
- Sentiment Tone: {text_metrics['sentiment_label']} (Polarity: {text_metrics['sentiment_polarity']}, Subjectivity: {text_metrics['sentiment_subjectivity']})
- Flesch Readability Ease: {text_metrics['readability_ease']}/100 (Higher is simpler/clearer)
- Flesch Reading Grade Level: Grade {text_metrics['readability_grade']}
- Average Sentence Length: {text_metrics['avg_sentence_length']} words

---

### INSTRUCTIONS:
Generate a structured, professional, and friendly evaluation in **JSON format only**. Do not include backticks, markdown markers, or text outside of the JSON block.
Ensure all feedback is specific, deeply encouraging yet honest, and includes strengths, weaknesses, and clear actionable improvements for:
1. Language Evaluation
2. Speech Delivery Evaluation
3. Body Language Evaluation

Ensure the JSON matches this structure EXACTLY:
{{
  "overall_score": 85,
  "overall_summary": "Friendly, overarching summary of the interview performance.",
  "language_evaluation": {{
    "summary": "Friendly analysis of sentence flow, sentiment, reading depth, and confidence markers.",
    "strengths": ["Strength point 1", "Strength point 2"],
    "weaknesses": ["Weakness point 1", "Weakness point 2"],
    "to_improve": ["Actionable improvement tip 1", "Actionable improvement tip 2"],
    "sentence_corrections": [
      {{
        "original": "Exact problematic sentence from the transcript (include ALL sentences with filler words, poor structure, passive voice, or weak phrasing)",
        "better_alternative": "Polished, confident, professional and executive-level rephrasing",
        "explanation": "A brief, friendly 1-sentence reason why this version is stronger (e.g. removes filler, uses active voice, projects confidence)"
      }}
    ]
  }},
  "speech_delivery_evaluation": {{
    "summary": "Friendly analysis of speech rate, pause intervals, filler word frequencies, and tone shifts.",
    "strengths": ["Strength point 1", "Strength point 2"],
    "weaknesses": ["Weakness point 1", "Weakness point 2"],
    "to_improve": ["Actionable improvement tip 1", "Actionable improvement tip 2"]
  }},
  "body_language_evaluation": {{
    "summary": "Friendly analysis of facial expressiveness (smiles), posture alignment, head stability, and eye contact simulation.",
    "strengths": ["Strength point 1", "Strength point 2"],
    "weaknesses": ["Weakness point 1", "Weakness point 2"],
    "to_improve": ["Actionable improvement tip 1", "Actionable improvement tip 2"],
    "timestamped_feedback": [
      {{
        "time": "0m 12s",
        "issue": "Briefly describe the issue based on the provided improvement timestamps (e.g. No smile detected or Face not visible).",
        "advice": "Friendly actionable advice to correct it."
      }}
    ]
  }}
}}

CRITICAL RULES:
1. For sentence_corrections, you MUST ONLY select sentences EXACTLY as they appear in the provided CANDIDATE TRANSCRIPT. DO NOT hallucinate or generate random conversational sentences. Include every exact sentence that contains filler words (um, uh, like, basically), poor structure, passive voice, or lacks confidence. If the transcript has no errors, leave sentence_corrections empty. Do not limit to 2-3 examples.
2. DO NOT mention or discuss filler words, speaking pace, or audio metrics in the 'language_evaluation' or 'body_language_evaluation' sections. Reserve ALL audio and filler feedback strictly for the 'speech_delivery_evaluation' section to avoid repetition.
"""

        logger.info(f"Sending completion request to OpenAI model: {model_to_use}")
        response = client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": "You are a professional, friendly, and structured interview analysis coach. You must return valid JSON strictly conforming to the requested schema."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=2500
        )

        raw_response = response.choices[0].message.content
        logger.info("Azure OpenAI Response received. Parsing JSON...")
        evaluation_report = json.loads(raw_response)
        
        # Append granular offline grammar corrections to the LLM coaching suggestions
        grammar_corrections = _get_grammar_corrections(transcript)
        if "language_evaluation" not in evaluation_report:
            evaluation_report["language_evaluation"] = {}
        if "sentence_corrections" not in evaluation_report["language_evaluation"]:
            evaluation_report["language_evaluation"]["sentence_corrections"] = []
            
        evaluation_report["language_evaluation"]["sentence_corrections"].extend(grammar_corrections)
        
        return evaluation_report

    except Exception as e:
        logger.error(f"Failed to generate report via Azure OpenAI: {str(e)}. Falling back to local generation.")
        return get_mock_ai_evaluation(transcript, audio_metrics, video_metrics, text_metrics)


def get_mock_ai_evaluation(transcript, audio_metrics, video_metrics, text_metrics):
    """
    Generates a high-quality, customized, interactive mock report 
    based directly on computed metrics. Safe for development.
    """
    logger.info("Running custom fallback report compiler...")
    
    # Base overall score calculation
    score = int(
        (video_metrics["head_stability_score"] * 0.2) +
        (video_metrics["eye_contact_score"] * 0.25) +
        (video_metrics["posture_score"] * 0.15) +
        (max(30, min(100, 100 - (audio_metrics["filler_words_count"] * 5))) * 0.2) +
        (max(40, min(100, 100 - abs(audio_metrics["wpm"] - 130) * 0.5)) * 0.2)
    )
    score = max(50, min(98, score))

    # Dynamic evaluations based on pace
    wpm_feedback = "Your speaking pace is at an ideal professional rhythm (110-150 WPM)."
    if audio_metrics["wpm"] > 150:
        wpm_feedback = "Your speed is slightly high (over 150 WPM), which might make it difficult for the interviewer to follow complex ideas."
    elif audio_metrics["wpm"] < 100:
        wpm_feedback = "Your speed is slightly measured or slow (under 100 WPM), suggesting cautious articulation but potentially losing listener momentum."

    # Dynamic evaluations based on fillers
    fillers_count = audio_metrics["filler_words_count"]
    filler_feedback = "You managed your voice exceptionally well, using very few verbal fillers."
    if fillers_count > 5:
        filler_feedback = f"You used {fillers_count} filler words (like 'um', 'like', or 'you know'). Minimizing these will make your speaking sound significantly more authoritative."

    # Dynamic evaluations based on smile
    smile_pct = video_metrics["smile_frequency_pct"]
    smile_feedback = "You showed outstanding warmth, maintaining a welcoming smile during the discussion."
    if smile_pct < 15:
        smile_feedback = "Your expression remained quite neutral. Injecting occasional smiles at key positive moments (e.g., introductions or accomplishments) will build much stronger rapport."

    # Dynamic evaluations based on stability
    stability = video_metrics["head_stability_score"]
    stability_feedback = "Your head movements were calm and balanced, indicating absolute poise and professional control."
    if stability < 70:
        stability_feedback = "There was noticeable movement or leaning. Stabilizing your body posture will make you project much greater executive presence."

    # Construct complete mock JSON matching the requested API format
    report = {
        "overall_score": score,
        "overall_summary": (
            f"You delivered a strong practice interview with an overall score of {score}%. "
            f"Your verbal delivery was clear, and you showed positive camera presence. "
            f"By refining a few elements—such as stabilizing your head posture and smoothing out transitions—you can "
            f"make your interview presence feel truly remarkable."
        ),
        "language_evaluation": {
            "summary": (
                f"Linguistically, your speaking style was rated as having '{text_metrics['sentiment_label']}' tone. "
                f"Your sentences averaged {text_metrics['avg_sentence_length']} words, which represents an excellent level "
                f"of conversational structure. Readability scored at {text_metrics['readability_ease']}/100, which means your ideas are "
                f"highly accessible and direct."
            ),
            "strengths": [
                "Excellent vocabulary and choice of clear, professional sentence structures.",
                f"Kept an engaging conversation depth with an average sentence size of {text_metrics['avg_sentence_length']} words."
            ],
            "weaknesses": [
                "Some complex thoughts could be broken down further to prevent long run-on sentences.",
                "Occasional conversational markers could be replaced with decisive, outcome-focused language."
            ],
            "to_improve": [
                "Practice using active verbs and bulleted-style verbal descriptions (e.g., 'First, I did X; Second, the result was Y').",
                "Inject structured pausing after major statements instead of merging sentences together."
            ],
            "sentence_corrections": []
        },
        "speech_delivery_evaluation": {
            "summary": (
                f"You spoke at an average pace of {audio_metrics['wpm']} words per minute. {wpm_feedback} "
                f"You registered {audio_metrics['pause_count']} significant pauses. {filler_feedback}"
            ),
            "strengths": [
                f"Maintained an average tempo of {audio_metrics['wpm']} WPM which aligns perfectly with modern listening standards.",
                f"Showed great vocal expression with a pitch variance of {audio_metrics['pitch_variability']} Hz, avoiding monotone tendencies."
            ],
            "weaknesses": [
                f"Recorded a total of {audio_metrics['filler_words_count']} filler words, specifically relying on conversational crutches.",
                f"Silence segments comprised {audio_metrics['silence_percentage']}% of the audio; slightly high pauses can disrupt fluency."
            ],
            "to_improve": [
                "When you feel a filler word coming on, swallow the word and take a silent, brief breath instead.",
                "Practice timing your answers to be under 2 minutes to keep WPM rhythm consistent and dynamic."
            ]
        },
        "body_language_evaluation": {
            "summary": (
                f"Your visual presence was very professional. Your face was visible for {video_metrics['face_visibility_pct']}% of the frames, "
                f"and you maintained an eye contact score of {video_metrics['eye_contact_score']}/100. {smile_feedback} {stability_feedback}"
            ),
            "strengths": [
                f"High camera presence and frame alignment, remaining in the frame for {video_metrics['face_visibility_pct']}% of the time.",
                f"Excellent eye level simulated contact ({video_metrics['eye_contact_score']}/100), keeping your eyes directed forward."
            ],
            "weaknesses": [
                f"Your head stability score was {video_metrics['head_stability_score']}/100, showing minor swaying or camera posture shifts.",
                f"Smile frequency sat around {video_metrics['smile_frequency_pct']}%, representing a slightly tense or overly serious presentation."
            ],
            "to_improve": [
                "Position your web camera exactly at eye level. This naturally makes simulated eye contact look warmer and more natural.",
                "Deliberately smile during the first 30 seconds of introducing yourself to instantly set a friendly tone."
            ],
            "timestamped_feedback": [
                {
                    "time": "0m 08s",
                    "issue": "No smile detected",
                    "advice": "Try to maintain a warm smile to build rapport."
                },
                {
                    "time": "0m 14s",
                    "issue": "Face not fully visible",
                    "advice": "Ensure you are centered within the camera frame."
                }
            ]
        }
    }
    
    import re
    # Dynamically extract real sentences with fillers from the transcript for the mock
    import re
    try:
        import language_tool_python
    except ImportError:
        language_tool_python = None

    # Add harsh pitch timestamps to body language evaluation
    harsh_pitches = audio_metrics.get("harsh_pitch_timestamps", [])
    if harsh_pitches:
        if "timestamped_feedback" not in report["body_language_evaluation"]:
            report["body_language_evaluation"]["timestamped_feedback"] = []
        for time_stamp in harsh_pitches:
            report["body_language_evaluation"]["timestamped_feedback"].append({
                "time": time_stamp,
                "issue": "Harsh / Spiking Pitch Tone",
                "advice": "Your voice pitch spiked significantly here. Try to take a deep breath and lower your tone to project calm confidence."
            })

    report["language_evaluation"]["sentence_corrections"] = _get_grammar_corrections(transcript)
    return report

def _get_grammar_corrections(transcript):
    import re
    try:
        import language_tool_python
    except ImportError:
        language_tool_python = None

    sentences = re.split(r'(?<=[.!?]) +', transcript)
    
    # Run-on breaker
    cleaned_sentences = []
    for s in sentences:
        words = s.split()
        if len(words) > 30:
            s_clean = re.sub(r'\s+and\s+', '. ', s, flags=re.IGNORECASE)
            s_clean = re.sub(r'\s+then\s+', '. ', s_clean, flags=re.IGNORECASE)
            s_clean = re.sub(r'\s+but\s+', '. ', s_clean, flags=re.IGNORECASE)
            for sub_s in s_clean.split('. '):
                if sub_s.strip():
                    cleaned_sentences.append(sub_s.strip().capitalize() + '.')
        else:
            cleaned_sentences.append(s)

    corrections = []
    if language_tool_python:
        try:
            tool = language_tool_python.LanguageTool('en-US')
            for sentence in cleaned_sentences:
                if len(sentence) < 10:
                    continue
                matches = tool.check(sentence)
                if matches:
                    better = sentence
                    # Apply replacements from last to first to preserve offsets
                    explanations = []
                    for match in reversed(matches):
                        if match.replacements:
                            better = better[:match.offset] + match.replacements[0] + better[match.offset + match.error_length:]
                            explanations.append(match.message)
                    
                    if better != sentence:
                        # Capitalize standalone "I"
                        better = re.sub(r'\bi\b', 'I', better)
                        
                        corrections.append({
                            "original": sentence.strip(),
                            "better_alternative": better,
                            "explanation": " | ".join(set(explanations))
                        })
                        if len(corrections) >= 25:
                            break
            tool.close()
        except Exception as e:
            logger.error(f"Language tool failed: {e}")

    # Fallback to filler removal if no grammar errors found
    if not corrections:
        fillers = ['um', 'uh', 'like', 'basically', 'actually', 'you know']
        for sentence in cleaned_sentences:
            sentence_lower = sentence.lower()
            if any(f in sentence_lower for f in fillers):
                better = sentence
                for filler in fillers:
                    better = re.sub(rf'\b{filler}\b[,\s]*', '', better, flags=re.IGNORECASE)
                better = re.sub(r'\bi\b', 'I', better)
                if better != sentence:
                    corrections.append({
                        "original": sentence.strip(),
                        "better_alternative": better.strip().capitalize(),
                        "explanation": "Removed conversational filler words to make the statement more professional."
                    })
                    if len(corrections) >= 25:
                        break

    if not corrections and cleaned_sentences:
        first_sent = cleaned_sentences[0].strip()
        corrections.append({
            "original": first_sent,
            "better_alternative": first_sent.capitalize() + " (Polished)",
            "explanation": "Ensure strong opening statements to build confidence."
        })

    return corrections

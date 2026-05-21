import os
import json
import logging
from textblob import TextBlob
import textstat
from openai import AzureOpenAI

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
    # Check Azure credentials
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    if not api_key or not endpoint:
        logger.warning("Azure OpenAI credentials not set. Falling back to offline report generation.")
        return get_mock_ai_evaluation(audio_metrics, video_metrics, text_metrics)

    try:
        logger.info("Initializing Azure OpenAI client...")
        client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )

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
    "to_improve": ["Actionable improvement tip 1", "Actionable improvement tip 2"]
  }}
}}

CRITICAL: For sentence_corrections, scan the ENTIRE transcript comprehensively. Include EVERY sentence that contains filler words (um, uh, like, basically, you know, actually), passive voice, weak openers ("I think", "I feel like", "I just"), overly long run-ons, or lacks confidence. Do not limit to 2-3 examples.
"""

        logger.info(f"Sending completion request to Azure OpenAI model deployment: {deployment_name}")
        response = client.chat.completions.create(
            model=deployment_name,
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
        return evaluation_report

    except Exception as e:
        logger.error(f"Failed to generate report via Azure OpenAI: {str(e)}. Falling back to local generation.")
        return get_mock_ai_evaluation(audio_metrics, video_metrics, text_metrics)


def get_mock_ai_evaluation(audio_metrics, video_metrics, text_metrics):
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
            "sentence_corrections": [
                {
                    "original": "Um, my name is Alex. Actually, I am very excited to, you know, talk about my experience.",
                    "better_alternative": "Hello, I'm Alex, and I am delighted to discuss my professional background.",
                    "explanation": "Replaces hesitant fillers ('um', 'actually', 'you know') with a direct, positive, and structured opening statement."
                },
                {
                    "original": "I have worked in software engineering for three years, and basically, my focus has been on building scalable web apps.",
                    "better_alternative": "Over the past three years as a software engineer, my primary focus has been developing highly scalable web applications.",
                    "explanation": "Removes the word 'basically' (which can sound minimizing) and constructs a more active, high-impact phrasing."
                },
                {
                    "original": "I think one of my, uh, greatest strengths is problem solving, though sometimes I can get, you know, a bit caught up in the details.",
                    "better_alternative": "One of my key strengths is collaborative problem solving. I pride myself on maintaining high detail quality while keeping broad project objectives in view.",
                    "explanation": "Eliminates visual filler tokens ('uh', 'I think', 'you know') and reframes a minor weakness into a balanced professional asset."
                }
            ]
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
            ]
        }
    }
    
    return report

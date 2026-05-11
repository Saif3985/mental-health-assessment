"""
═══════════════════════════════════════════════════════════════════════════════
                    MENTAL HEALTH ASSESSMENT SYSTEM
                        Complete All-in-One Solution
═══════════════════════════════════════════════════════════════════════════════

Description:
    A comprehensive mental health assessment tool with voice support and 
    beautiful Gradio web interface. Combines psychological evaluation with 
    the standardized PHQ-9 depression screening questionnaire.

Features:
    • 10 psychological metric questions (depression, anxiety, stress, etc.)
    • PHQ-9 depression screening (9 standardized questions)
    • Voice input support via Whisper AI
    • Text-to-speech output
    • Real-time conversation history tracking
    • Automatic scoring and risk assessment
    • Beautiful dark-themed Gradio GUI
    • Local data storage (privacy-focused)
    • Detailed JSON reports with timestamps

Author: Mental Health Assessment Team
Version: 1.0
Last Updated: May 2026
License: Educational and Research Use

═══════════════════════════════════════════════════════════════════════════════
"""

# ═══════════════════════════════════════════════════════════════════════════
#                               LIBRARY IMPORTS
# ═══════════════════════════════════════════════════════════════════════════

# Standard library imports for core Python functionality
import json          # For reading/writing JSON report files
import random        # For shuffling question order
import os           # For file system operations
import re           # For regex pattern matching in LLM output parsing
from datetime import datetime    # For timestamps in reports
from pathlib import Path        # For modern file path handling

# Data processing library
import pandas as pd  # For creating CSV files with assessment scores

# ─────────────────────────────────────────────────────────────────────────────
# Voice Processing Libraries (Optional - graceful degradation if missing)
# ─────────────────────────────────────────────────────────────────────────────

VOICE_AVAILABLE = False  # Global flag to track voice capability
try:
    import sounddevice as sd     # For microphone audio recording
    import soundfile as sf       # For saving/loading audio files
    import whisper              # OpenAI Whisper for speech-to-text
    import pyttsx3             # Text-to-speech engine
    VOICE_AVAILABLE = True     # Set flag if all imports succeed
except ImportError:
    # If any voice library is missing, the system falls back to text-only mode
    # This allows the application to run even without voice capabilities
    pass

# ─────────────────────────────────────────────────────────────────────────────
# LLM Processing Libraries (Optional - falls back to rule-based if missing)
# ─────────────────────────────────────────────────────────────────────────────

LLM_AVAILABLE = False  # Global flag to track LLM capability
try:
    import torch                                      # PyTorch deep learning framework
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from langchain_huggingface import HuggingFacePipeline
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    LLM_AVAILABLE = True  # Set flag if all LLM libraries load successfully
except ImportError:
    # If LLM libraries are missing, system uses rule-based scoring
    # This allows basic functionality without requiring heavy AI models
    pass

# ─────────────────────────────────────────────────────────────────────────────
# GUI Framework Library (Optional - CLI fallback available)
# ─────────────────────────────────────────────────────────────────────────────

GUI_AVAILABLE = False  # Global flag to track Gradio availability
try:
    import gradio as gr  # Modern web UI framework for ML applications
    GUI_AVAILABLE = True
except ImportError:
    # If Gradio is not installed, the system provides a command-line interface
    pass


# ═══════════════════════════════════════════════════════════════════════════
#                          MAIN APPLICATION CLASS
# ═══════════════════════════════════════════════════════════════════════════

class MentalHealthAssistant:
    """
    Core mental health assessment system that handles question flow,
    response collection, scoring, and report generation.
    
    This class manages the complete assessment lifecycle from welcome
    to final results, supporting both text and voice interaction modes.
    """
    
    def __init__(self, use_llm=True, model_path=None):
        """
        Initialize the assessment system with all questions, settings,
        and prepare data structures for storing responses.
        
        Args:
            use_llm (bool): Whether to use Llama 3B for intelligent scoring (default: True)
            model_path (str): Path to local Llama model, or None to auto-detect
        """
        
        # ─────────────────────────────────────────────────────────────────────
        # File System Setup
        # ─────────────────────────────────────────────────────────────────────
        
        # Create directory for storing assessment results if it doesn't exist
        self.results_dir = Path("assessment_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # ─────────────────────────────────────────────────────────────────────
        # Voice System Components (initialized on demand)
        # ─────────────────────────────────────────────────────────────────────
        
        self.whisper_model = None  # Whisper AI model for speech recognition
        self.tts_engine = None     # Text-to-speech engine instance
        
        # ─────────────────────────────────────────────────────────────────────
        # LLM System Components (initialized on demand)
        # ─────────────────────────────────────────────────────────────────────
        
        self.use_llm = use_llm and LLM_AVAILABLE  # Only use LLM if requested and available
        self.llm_model = None      # Llama language model for intelligent scoring
        self.llm_tokenizer = None  # Tokenizer for processing text input
        self.llm_chain = None      # LangChain pipeline for structured prompting
        self.model_path = model_path or "./"  # Default to current directory
        
        # Try to initialize LLM if requested
        if self.use_llm:
            print("🤖 Initializing Llama 3B model for intelligent response scoring...")
            llm_success = self.init_llm()
            if llm_success:
                print("✅ Llama 3B model loaded successfully!")
            else:
                print("⚠️  Llama model failed to load. Falling back to rule-based scoring.")
                self.use_llm = False
        
        # ─────────────────────────────────────────────────────────────────────
        # Assessment State Variables
        # ─────────────────────────────────────────────────────────────────────
        
        self.current_question = 0   # Track which question we're on
        self.responses = {}         # Store all user answers {metric: answer}
        self.scores = {}           # Store calculated scores {metric: score}
        
        # ─────────────────────────────────────────────────────────────────────
        # Psychological Assessment Questions (10 metrics)
        # ─────────────────────────────────────────────────────────────────────
        
        # These questions evaluate key psychological dimensions
        # Each question maps to a specific mental health metric
        self.questions = {
            "depression": "Have you found yourself feeling hopeless about what lies ahead lately?",
            "aggression": "Have you noticed feeling irritable or snapping at people more quickly?",
            "sadness": "Have there been moments where sadness weighed you down?",
            "concentration": "Would you say it has been easy to focus, or have tasks felt harder to complete?",
            "tiredness": "How often has low energy or fatigue gotten in the way of your day?",
            "suicidal": "Have you had any thoughts about not wanting to be here anymore?",
            "neuroticism": "Have there been times when stress felt too heavy to manage?",
            "movingon": "How easy has it been for you to let go of things that bothered you in the past?",
            "overthinking": "Do you ever feel stuck in your head, replaying things over and over?",
            "moodswings": "Have your emotions felt unpredictable—like shifting quickly from calm to upset?"
        }
        
        # ─────────────────────────────────────────────────────────────────────
        # PHQ-9 Depression Screening Questions (9 standardized items)
        # ─────────────────────────────────────────────────────────────────────
        
        # PHQ-9 is a validated clinical tool for depression screening
        # Used by healthcare professionals worldwide
        self.phq9_questions = {
            "Interest Loss": "Little interest or pleasure in doing things?",
            "Depressed Mood": "Feeling down, depressed, or hopeless?",
            "Sleep Issues": "Trouble falling asleep, staying asleep, or sleeping too much?",
            "Fatigue": "Feeling tired or having little energy?",
            "Appetite Issues": "Poor appetite or overeating?",
            "Self-Worth": "Feeling bad about yourself, or that you're a failure?",
            "Concentration Issues": "Trouble concentrating on things?",
            "Motor Function": "Moving or speaking slowly, or being restless?",
            "Suicidal Thoughts": "Thoughts of hurting yourself or being better off dead?"
        }
    
    # ═════════════════════════════════════════════════════════════════════════
    #                      LLM INITIALIZATION AND SETUP
    # ═════════════════════════════════════════════════════════════════════════
    
    def init_llm(self):
        """
        Initialize the Llama 3.2-3B-Instruct model for intelligent response scoring.
        
        This method loads the language model, creates the tokenizer, and sets up
        a LangChain pipeline with a carefully crafted prompt template that instructs
        the model to analyze psychological responses and assign severity scores.
        
        Returns:
            bool: True if LLM initialized successfully, False otherwise
        """
        
        if not LLM_AVAILABLE:
            return False
        
        try:
            # ─────────────────────────────────────────────────────────────────
            # Load Llama 3B Model and Tokenizer
            # ─────────────────────────────────────────────────────────────────
            
            print(f"   Loading model from: {self.model_path}")
            
            # Load the tokenizer (converts text to numbers the model understands)
            self.llm_tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Load the actual language model
            # Check if GPU is available for faster processing
            if torch.cuda.is_available():
                print("   Using GPU acceleration (CUDA)")
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map="auto",  # Automatically distribute across available GPUs
                    torch_dtype=torch.float16  # Use half-precision for speed
                )
            else:
                print("   Using CPU (this will be slower)")
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float32,  # Full precision on CPU
                    low_cpu_mem_usage=True
                )
            
            # ─────────────────────────────────────────────────────────────────
            # Create the Prompt Template for Response Scoring
            # ─────────────────────────────────────────────────────────────────
            
            # This template teaches the model exactly how to analyze responses
            prompt_template = PromptTemplate(
                input_variables=["question", "answer", "metric"],
                template="""
You are an expert clinical psychologist analyzing patient responses to mental health screening questions.

Your task: Evaluate the severity of the response and assign a score between 0.0 and 1.0 for the given psychological metric.

Scoring Guidelines:
- 0.0 to 0.2: No concern / Very positive response
- 0.3 to 0.4: Minimal concern / Mild symptoms
- 0.5 to 0.6: Moderate concern / Notable symptoms
- 0.7 to 0.8: High concern / Significant symptoms
- 0.9 to 1.0: Severe concern / Crisis level

Consider:
- Frequency words (never, rarely, sometimes, often, always)
- Intensity words (a little, somewhat, very, extremely)
- Coping indicators (manageable, overwhelming, can't handle)
- Emotional language and distress markers

Examples:

<example>
metric: "anxiety"
question: "Do you feel anxious for prolonged periods of time?"
answer: "Yes. I often feel like I am being drowned by anxiety. I cannot control it."
output: {{"anxiety": 0.95}}
</example>

<example>
metric: "anxiety"
question: "Do you feel anxious for prolonged periods of time?"
answer: "Not really. I do feel stress but I overcome it without issues."
output: {{"anxiety": 0.15}}
</example>

<example>
metric: "depression"
question: "Do you feel down a lot of the time?"
answer: "Sometimes I feel a bit low, but it passes after a day or two."
output: {{"depression": 0.35}}
</example>

Now analyze this response:

metric: {metric}
question: {question}
answer: {answer}

Provide ONLY the JSON output in this exact format:
<output>
{{"{metric}": score}}
</output>
"""
            )
            
            # ─────────────────────────────────────────────────────────────────
            # Create the Text Generation Pipeline
            # ─────────────────────────────────────────────────────────────────
            
            # Configure the model for consistent, focused responses
            text_generation_pipeline = pipeline(
                "text-generation",
                model=self.llm_model,
                tokenizer=self.llm_tokenizer,
                max_new_tokens=150,      # Limit response length (we only need a score)
                do_sample=False,         # Deterministic output (no randomness)
                temperature=0.1,         # Low temperature = more focused responses
                top_p=0.90              # Nucleus sampling for quality
            )
            
            # Wrap the pipeline in LangChain's HuggingFacePipeline
            llm_wrapper = HuggingFacePipeline(pipeline=text_generation_pipeline)
            
            # ─────────────────────────────────────────────────────────────────
            # Create the Complete LLM Chain
            # ─────────────────────────────────────────────────────────────────
            
            # This chain combines the prompt template with the model
            self.llm_chain = LLMChain(
                llm=llm_wrapper,
                prompt=prompt_template
            )
            
            return True
            
        except Exception as e:
            print(f"   Error during LLM initialization: {e}")
            return False
    
    def score_with_llm(self, question, answer, metric):
        """
        Use Llama 3B to intelligently score a psychological response.
        
        The model reads the question and answer, considers the psychological
        context, and assigns an appropriate severity score based on clinical
        understanding of mental health symptoms.
        
        Args:
            question (str): The psychological question that was asked
            answer (str): The user's response to the question
            metric (str): The psychological metric being evaluated
            
        Returns:
            float: Score between 0.0 and 1.0, or 0.5 if scoring fails
        """
        
        if not self.llm_chain:
            # If LLM is not available, fall back to rule-based scoring
            return self.score_answer(answer)
        
        try:
            # ─────────────────────────────────────────────────────────────────
            # Generate LLM Response
            # ─────────────────────────────────────────────────────────────────
            
            # Invoke the LLM chain with the question, answer, and metric
            llm_output = self.llm_chain.invoke({
                "question": question,
                "answer": answer,
                "metric": metric
            })
            
            # ─────────────────────────────────────────────────────────────────
            # Extract and Parse the Score
            # ─────────────────────────────────────────────────────────────────
            
            # The LLM output is in llm_output['text']
            output_text = llm_output['text']
            
            # Try to find the JSON output in <output> tags
            match = re.findall(r"<output>(.*?)</output>", output_text, re.DOTALL)
            
            if match:
                json_str = match[-1].strip()
            else:
                # Fallback: try to find any JSON object in the response
                json_match = re.search(r'\{[^}]+\}', output_text)
                if json_match:
                    json_str = json_match.group()
                else:
                    print(f"⚠️  Could not parse LLM output, using fallback score")
                    return 0.5
            
            # Parse the JSON to extract the score
            score_dict = json.loads(json_str)
            score = score_dict.get(metric, 0.5)
            
            # Ensure score is within valid range
            score = max(0.0, min(1.0, float(score)))
            
            return score
            
        except Exception as e:
            print(f"⚠️  LLM scoring error: {e}. Using rule-based fallback.")
            return self.score_answer(answer)
    
    # ═════════════════════════════════════════════════════════════════════════
    #                           VOICE PROCESSING METHODS
    # ═════════════════════════════════════════════════════════════════════════
    
    def init_voice(self):
        """
        Initialize voice processing components (TTS and STT).
        
        Returns:
            bool: True if voice components initialized successfully, False otherwise
        """
        
        # Check if voice libraries are available
        if not VOICE_AVAILABLE:
            return False
        
        try:
            # ─────────────────────────────────────────────────────────────────
            # Initialize Text-to-Speech Engine
            # ─────────────────────────────────────────────────────────────────
            
            self.tts_engine = pyttsx3.init()
            
            # Configure speech rate (words per minute)
            self.tts_engine.setProperty('rate', 130)  # Natural speaking pace
            
            # Configure volume level (0.0 to 1.0)
            self.tts_engine.setProperty('volume', 0.8)  # Slightly below maximum
            
            # Try to use a female voice if available (index 1 is typically female)
            voices = self.tts_engine.getProperty('voices')
            if len(voices) > 1:
                self.tts_engine.setProperty('voice', voices[1].id)
            
            # ─────────────────────────────────────────────────────────────────
            # Initialize Speech Recognition (Whisper AI)
            # ─────────────────────────────────────────────────────────────────
            
            # Load the "tiny" model for fast processing (39MB)
            # Other options: base (74MB), small (244MB), medium (769MB), large (1550MB)
            # Tiny is sufficient for clear speech and processes in 1-2 seconds
            if self.whisper_model is None:
                self.whisper_model = whisper.load_model("tiny")
            
            return True
            
        except Exception as e:
            # If initialization fails, voice mode will be disabled
            print(f"Voice initialization failed: {e}")
            return False
    
    def speak(self, text):
        """
        Convert text to speech and play it through speakers.
        
        Args:
            text (str): The text to speak aloud
        """
        if self.tts_engine:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()  # Block until speech is complete
    
    def transcribe_audio(self, audio_file):
        """
        Convert recorded audio file to text using Whisper AI.
        
        Args:
            audio_file (str): Path to the audio file to transcribe
            
        Returns:
            str: Transcribed text from the audio
        """
        
        # Load Whisper model if not already loaded
        if self.whisper_model is None:
            self.whisper_model = whisper.load_model("tiny")
        
        # Transcribe the audio file
        # Whisper automatically handles different audio formats and quality levels
        result = self.whisper_model.transcribe(audio_file)
        
        # Extract and clean the transcribed text
        return result["text"].strip()
    
    # ═════════════════════════════════════════════════════════════════════════
    #                          SCORING AND ANALYSIS METHODS
    # ═════════════════════════════════════════════════════════════════════════
    
    def score_answer(self, answer):
        """
        Analyze a response and calculate a mental health score (0.0 to 1.0).
        
        This rule-based scoring system evaluates the sentiment and severity
        of responses. Higher scores indicate greater concern.
        
        Score Interpretation:
            0.0 - 0.3: Minimal concern (healthy response)
            0.3 - 0.5: Low concern (some mild symptoms)
            0.5 - 0.7: Moderate concern (notable symptoms)
            0.7 - 1.0: High concern (significant symptoms)
        
        Args:
            answer (str): The user's response text
            
        Returns:
            float: Score between 0.0 and 1.0
        """
        
        # Convert to lowercase for case-insensitive matching
        answer_lower = answer.lower()
        
        # ─────────────────────────────────────────────────────────────────────
        # Define Keywords for Sentiment Analysis
        # ─────────────────────────────────────────────────────────────────────
        
        # Negative indicators suggest concerning mental health symptoms
        negative = [
            'yes', 'often', 'always', 'very', 'extremely', 'terrible', 
            'hopeless', 'overwhelming', 'unable', 'difficult', 'constantly',
            'severe', 'helpless', 'can\'t', 'never been worse', 'awful'
        ]
        
        # Positive indicators suggest healthy coping or minimal symptoms
        positive = [
            'no', 'never', 'rarely', 'sometimes', 'manageable', 'okay', 
            'fine', 'good', 'better', 'well', 'improving', 'handling it',
            'under control', 'not really'
        ]
        
        # ─────────────────────────────────────────────────────────────────────
        # Count Keyword Occurrences
        # ─────────────────────────────────────────────────────────────────────
        
        # Count how many negative keywords appear in the response
        neg_count = sum(1 for word in negative if word in answer_lower)
        
        # Count how many positive keywords appear in the response
        pos_count = sum(1 for word in positive if word in answer_lower)
        
        # ─────────────────────────────────────────────────────────────────────
        # Calculate Score Based on Keyword Balance
        # ─────────────────────────────────────────────────────────────────────
        
        if neg_count > pos_count:
            # More negative indicators: score above 0.5 (concern level)
            # Each additional negative word increases score by 0.15
            score = min(0.5 + (neg_count * 0.15), 1.0)  # Cap at 1.0
            
        elif pos_count > neg_count:
            # More positive indicators: score below 0.5 (healthier)
            # Each positive word decreases score by 0.15
            score = max(0.5 - (pos_count * 0.15), 0.0)  # Floor at 0.0
            
        else:
            # Equal or no clear indicators: neutral score
            score = 0.5
        
        return score
    
    def calculate_phq9(self, responses):
        """
        Calculate PHQ-9 depression score and determine severity level.
        
        The PHQ-9 is a validated clinical tool that scores each response:
            Never = 0 points
            Sometimes = 1 point
            Often = 2 points
        
        Total score ranges from 0-27 with the following interpretation:
            0-4: Minimal or no depression
            5-9: Mild depression
            10-14: Moderate depression
            15-19: Moderately severe depression
            20-27: Severe depression
        
        Args:
            responses (dict): Dictionary of PHQ-9 responses
            
        Returns:
            tuple: (score, severity_level, clinical_advice)
        """
        
        # Define the scoring map
        score_map = {
            "never": 0,
            "sometimes": 1,
            "often": 2
        }
        
        # Calculate total score by summing all responses
        total = 0
        for response in responses.values():
            resp_lower = response.lower()
            
            # Match response to score value
            for key, value in score_map.items():
                if key in resp_lower:
                    total += value
                    break
        
        # ─────────────────────────────────────────────────────────────────────
        # Determine Severity Level and Provide Clinical Guidance
        # ─────────────────────────────────────────────────────────────────────
        
        if total <= 4:
            return total, "Minimal or None", "Keep up self-care! 🌿"
            
        elif total <= 9:
            return total, "Mild Depression", "Consider talking to someone you trust 🤝"
            
        elif total <= 14:
            return total, "Moderate Depression", "Seeking support from a therapist could help 💙"
            
        elif total <= 19:
            return total, "Moderately Severe", "Please reach out for professional help ❤️"
            
        else:
            return total, "Severe Depression", "Seek professional help immediately ❤️"
    
    # ═════════════════════════════════════════════════════════════════════════
    #                          REPORT GENERATION
    # ═════════════════════════════════════════════════════════════════════════
    
    def generate_report(self):
        """
        Create comprehensive assessment report with scores and recommendations.
        
        Generates both a detailed JSON file and a CSV file with scores.
        All files are timestamped and saved locally for privacy.
        
        Returns:
            tuple: (report_dict, report_file_path)
        """
        
        # ─────────────────────────────────────────────────────────────────────
        # Calculate Overall Metrics
        # ─────────────────────────────────────────────────────────────────────
        
        # Calculate average score across all psychological metrics
        avg_score = sum(self.scores.values()) / len(self.scores) if self.scores else 0
        
        # Identify metrics with concerning scores (> 0.7)
        high_risk = [k for k, v in self.scores.items() if v > 0.7]
        
        # ─────────────────────────────────────────────────────────────────────
        # Determine Overall Risk Level
        # ─────────────────────────────────────────────────────────────────────
        
        if avg_score >= 0.7 or 'suicidal' in high_risk:
            # Critical risk level requires immediate attention
            risk = "High Risk - Seek professional help immediately"
            
        elif avg_score >= 0.5 or len(high_risk) >= 3:
            # Moderate risk suggests professional consultation
            risk = "Moderate Risk - Consider mental health professional"
            
        elif avg_score >= 0.3:
            # Low risk but monitoring recommended
            risk = "Low Risk - Monitor and practice self-care"
            
        else:
            # Minimal risk, healthy functioning
            risk = "Minimal Risk - Continue healthy habits"
        
        # ─────────────────────────────────────────────────────────────────────
        # Create Report Data Structure
        # ─────────────────────────────────────────────────────────────────────
        
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report = {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'scores': self.scores,
            'responses': self.responses,
            'average_score': avg_score,
            'high_risk_areas': high_risk,
            'risk_level': risk
        }
        
        # ─────────────────────────────────────────────────────────────────────
        # Save Report as JSON (detailed format)
        # ─────────────────────────────────────────────────────────────────────
        
        report_file = self.results_dir / f"assessment_{session_id}.json"
        with open(report_file, 'w') as f:
            json.dump(report, indent=2, fp=f)
        
        return report, str(report_file)


# ═══════════════════════════════════════════════════════════════════════════
#                         GRADIO WEB INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

def create_gui():
    """
    Create and configure the Gradio web interface for the assessment.
    
    This function builds a beautiful, interactive web UI with:
        - Modern dark theme with purple/indigo gradient
        - Two-column layout (questions left, history right)
        - Real-time conversation history tracking
        - Progress bar showing completion percentage
        - Voice and text input support
        - Responsive design that works on desktop and mobile
    
    Returns:
        gr.Blocks: Configured Gradio interface ready to launch
    """
    
    # ─────────────────────────────────────────────────────────────────────────
    # Initialize Assessment System
    # ─────────────────────────────────────────────────────────────────────────
    
    assistant = MentalHealthAssistant()
    
    # ─────────────────────────────────────────────────────────────────────────
    # State Variables for Question Flow Management
    # ─────────────────────────────────────────────────────────────────────────
    
    # These variables track where we are in the assessment process
    current_phase = "welcome"  # Phases: welcome → psychological → phq9 → complete
    question_list = list(assistant.questions.items())  # Convert dict to list for indexing
    phq9_list = list(assistant.phq9_questions.items())
    current_idx = 0  # Index of current question within current phase
    conversation_history = []  # Store all Q&A pairs for display
    
    # ─────────────────────────────────────────────────────────────────────────
    # Helper Function: Format Conversation History as HTML
    # ─────────────────────────────────────────────────────────────────────────
    
    def format_conversation_history():
        """
        Generate HTML markup for the conversation history sidebar.
        
        Creates a styled, scrollable list of all questions answered so far,
        with color-coded scores for psychological questions.
        
        Returns:
            str: HTML string with styled conversation history
        """
        if not conversation_history:
            return """
            <div class="history-container">
                <h3 style="margin-top: 0; color: #e0e7ff;">📝 Your Responses</h3>
                <p style="color: #94a3b8; font-size: 0.9em;">Your questions and answers will appear here as you progress.</p>
            </div>
            """
        
        history_html = """
        <div class="history-container">
            <h3 style="margin-top: 0; color: #e0e7ff;">📝 Your Responses ({} answered)</h3>
        """.format(len(conversation_history))
        
        for idx, item in enumerate(conversation_history, 1):
            # Determine emoji
            if idx == 1:
                emoji = "👋"
            elif idx <= 11:
                emoji = "🔹"
            else:
                emoji = "📋"
            
            # Format score
            score_html = ""
            if 'score' in item and item['score'] is not None:
                score_val = item['score']
                if score_val > 0.7:
                    score_color = "#ef4444"  # red
                    score_emoji = "🔴"
                elif score_val > 0.5:
                    score_color = "#f59e0b"  # orange
                    score_emoji = "🟡"
                else:
                    score_color = "#10b981"  # green
                    score_emoji = "🟢"
                
                score_html = f'<div style="color: {score_color}; font-size: 0.85em; margin-top: 8px;">{score_emoji} Score: {score_val:.2f}</div>'
            
            history_html += f"""
            <div class="history-item">
                <div class="history-question">{emoji} Q{idx}: {item['question']}</div>
                <div class="history-answer">💬 {item['answer']}</div>
                {score_html}
            </div>
            """
        
        history_html += "</div>"
        return history_html
    
    def calculate_progress():
        """Calculate overall progress percentage"""
        total_questions = 1 + len(question_list) + len(phq9_list)  # 1 welcome + 10 psych + 9 phq9 = 20
        answered = len(conversation_history)
        percentage = (answered / total_questions) * 100
        return percentage
    
    def process_answer(answer, audio):
        """Process text or voice answer"""
        nonlocal current_phase, current_idx
        
        # Handle voice input
        if audio is not None:
            if not VOICE_AVAILABLE:
                return (
                    "❌ Voice mode not available. Please type your answer.",
                    None,
                    gr.update(value=""),
                    format_conversation_history(),
                    gr.update(value=calculate_progress())
                )
            
            try:
                answer = assistant.transcribe_audio(audio)
            except Exception as e:
                return (
                    f"❌ Voice transcription failed: {e}\nPlease type your answer instead.",
                    None,
                    gr.update(value=""),
                    format_conversation_history(),
                    gr.update(value=calculate_progress())
                )
        
        if not answer or not answer.strip():
            return (
                "⚠️ Please provide an answer (text or voice).",
                None,
                gr.update(value=""),
                format_conversation_history(),
                gr.update(value=calculate_progress())
            )
        
        # Welcome phase
        if current_phase == "welcome":
            conversation_history.append({
                'question': 'How are you feeling today?',
                'answer': answer,
                'score': None
            })
            
            assistant.responses['initial_mood'] = answer
            current_phase = "psychological"
            current_idx = 0
            metric, question = question_list[0]
            
            next_question = f"""
## Psychological Assessment 🧠

**Question 1/10 - Progress: 5%**

### {question}

*Take your time and answer honestly. There are no right or wrong answers.*
"""
            
            return (
                next_question,
                None,
                gr.update(value=""),
                format_conversation_history(),
                gr.update(value=calculate_progress())
            )
        
        # Psychological questions
        elif current_phase == "psychological":
            metric, current_question = question_list[current_idx]
            
            # Use LLM for intelligent scoring if available, otherwise use rule-based
            if assistant.use_llm:
                score = assistant.score_with_llm(current_question, answer, metric)
            else:
                score = assistant.score_answer(answer)
            
            conversation_history.append({
                'question': current_question,
                'answer': answer,
                'score': score
            })
            
            assistant.responses[metric] = answer
            assistant.scores[metric] = score
            
            current_idx += 1
            
            if current_idx < len(question_list):
                metric, question = question_list[current_idx]
                progress_pct = 5 + (current_idx * 5)
                
                encouragement = ""
                if current_idx % 3 == 0:
                    encouragement = "### 💬 You're doing great! Keep going!\n\n"
                
                next_question = f"""
## Psychological Assessment 🧠

**Question {current_idx + 1}/10 - Progress: {progress_pct}%**

{encouragement}### {question}

*Take your time and answer honestly.*
"""
                
                return (
                    next_question,
                    None,
                    gr.update(value=""),
                    format_conversation_history(),
                    gr.update(value=calculate_progress())
                )
            else:
                current_phase = "phq9"
                current_idx = 0
                category, question = phq9_list[0]
                
                next_question = f"""
## PHQ-9 Depression Screening 📋

**Question 1/9 - Progress: 55%**

### {question}

**Please answer with:** Never, Sometimes, or Often

*This is a standardized depression screening tool.*
"""
                
                return (
                    next_question,
                    None,
                    gr.update(value=""),
                    format_conversation_history(),
                    gr.update(value=calculate_progress())
                )
        
        # PHQ-9 questions
        elif current_phase == "phq9":
            category, current_question = phq9_list[current_idx]
            
            conversation_history.append({
                'question': current_question,
                'answer': answer,
                'score': None
            })
            
            assistant.responses[f"phq9_{category}"] = answer
            
            current_idx += 1
            
            if current_idx < len(phq9_list):
                category, question = phq9_list[current_idx]
                progress_pct = 55 + (current_idx * 5)
                
                encouragement = ""
                if current_idx % 3 == 0:
                    encouragement = "### 💬 Almost done! Just a few more questions.\n\n"
                
                next_question = f"""
## PHQ-9 Depression Screening 📋

**Question {current_idx + 1}/9 - Progress: {progress_pct}%**

{encouragement}### {question}

**Please answer with:** Never, Sometimes, or Often
"""
                
                return (
                    next_question,
                    None,
                    gr.update(value=""),
                    format_conversation_history(),
                    gr.update(value=calculate_progress())
                )
            else:
                # Calculate results
                phq9_responses = {k.replace('phq9_', ''): v for k, v in assistant.responses.items() if k.startswith('phq9_')}
                phq9_score, severity, advice = assistant.calculate_phq9(phq9_responses)
                
                report, report_file = assistant.generate_report()
                
                # Add visual indicators for risk level
                risk_emoji = "🔴" if "High Risk" in report['risk_level'] else "🟡" if "Moderate Risk" in report['risk_level'] else "🟢"
                
                # Format high risk areas
                high_risk_display = ""
                if report['high_risk_areas']:
                    high_risk_display = "\n**⚠️ Areas Needing Attention:**\n"
                    for area in report['high_risk_areas']:
                        high_risk_display += f"- {area.replace('_', ' ').title()}\n"
                
                # Format results with simplified output - no extra sections
                results = f"""
# 🎉 Assessment Complete!

---

## 📊 Your Results

### Psychological Assessment Summary

**Average Score:** {report['average_score']:.2f} / 1.0
{high_risk_display}

**Score Guide:**
- 🟢 0.0-0.3: Minimal concern
- 🟡 0.3-0.7: Moderate concern  
- 🔴 0.7-1.0: High concern

---

### PHQ-9 Depression Screening Results

**Your Score:** {phq9_score} / 27

**Severity Level:** {severity}

**Recommendation:** {advice}

**PHQ-9 Score Guide:**
- 0-4: Minimal or none
- 5-9: Mild depression
- 10-14: Moderate depression
- 15-19: Moderately severe depression
- 20-27: Severe depression

---

### {risk_emoji} Overall Assessment

**{report['risk_level']}**

---

## 💾 Your Report

Your detailed assessment has been saved to:

📄 `{report_file}`
"""
                
                current_phase = "complete"
                return (
                    results,
                    None,
                    gr.update(value="", interactive=False),
                    format_conversation_history(),
                    gr.update(value=100)
                )
        
        return (
            "Assessment complete. Refresh the page to start a new assessment.",
            None,
            gr.update(value="", interactive=False),
            format_conversation_history(),
            gr.update(value=100)
        )
    
    # Create interface with custom CSS - Dark theme with better visibility
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .main-header {
        text-align: center;
        padding: 30px;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .history-container {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        max-height: 500px;
        overflow-y: auto;
        border: 2px solid #475569;
    }
    .history-item {
        background: rgba(99, 102, 241, 0.1);
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border-left: 4px solid #6366f1;
    }
    .history-question {
        color: #e0e7ff;
        font-weight: bold;
        margin-bottom: 8px;
    }
    .history-answer {
        color: #c7d2fe;
        margin-left: 10px;
    }
    .history-score {
        color: #fbbf24;
        font-size: 0.9em;
        margin-top: 5px;
    }
    """
    
    with gr.Blocks(title="Mental Health Assessment", theme=gr.themes.Base(), css=custom_css) as demo:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1 style="margin: 0; font-size: 2.5em;">🧠 Mental Health Assessment</h1>
            <p style="font-size: 1.2em; margin: 15px 0 5px 0; opacity: 0.95;">Complete psychological evaluation with voice support</p>
            <p style="font-size: 0.9em; opacity: 0.8;">19 questions • 10-15 minutes • Private & Secure</p>
        </div>
        """)
        
        # Progress bar
        progress_bar = gr.Slider(
            minimum=0,
            maximum=100,
            value=0,
            label="📊 Progress",
            interactive=False,
            show_label=True
        )
        
        # Main content area
        with gr.Row():
            # Left column - Current question and input
            with gr.Column(scale=6):
                question_display = gr.Markdown("""
## 👋 Welcome!

### How are you feeling today?

This assessment evaluates your mental well-being through psychological questions and the PHQ-9 depression screening.

**Answer by typing or using the microphone below.**

*All responses are private and stored only on your device.*
                """)
                
                text_input = gr.Textbox(
                    label="💬 Your Answer",
                    placeholder="Type your response here...",
                    lines=3
                )
                
                voice_input = gr.Audio(
                    label="🎤 Voice Answer (click to record)",
                    sources=["microphone"],
                    type="filepath"
                )
                
                submit_btn = gr.Button(
                    "Submit ➜",
                    variant="primary",
                    size="lg"
                )
            
            # Right column - Conversation history only
            with gr.Column(scale=4):
                conversation_display = gr.HTML(
                    """
                    <div class="history-container">
                        <h3 style="margin-top: 0; color: #e0e7ff;">📝 Your Responses</h3>
                        <p style="color: #94a3b8; font-size: 0.9em;">Your questions and answers will appear here as you progress through the assessment.</p>
                    </div>
                    """)
        
        # Connect the submit button
        submit_btn.click(
            fn=process_answer,
            inputs=[text_input, voice_input],
            outputs=[question_display, voice_input, text_input, conversation_display, progress_bar]
        )
    
    return demo


def main():
    """Main entry point"""
    
    print("\n" + "="*70)
    print(" " * 20 + "MENTAL HEALTH ASSESSMENT")
    print("="*70)
    
    if GUI_AVAILABLE:
        print("\n🎨 Launching GUI interface...")
        print("   Opening in your browser...")
        print("   Press Ctrl+C to stop\n")
        
        demo = create_gui()
        demo.launch(share=False, inbrowser=True)
    
    else:
        print("\n⚠️  Gradio not installed!")
        print("   Install with: pip install gradio")
        print("\n   Or use text-based interface:\n")
        
        # Fallback to command-line interface
        assistant = MentalHealthAssistant()
        
        print("Welcome! How are you feeling today?")
        initial = input("Your answer: ")
        assistant.responses['initial_mood'] = initial
        
        # Psychological questions
        questions_list = list(assistant.questions.items())
        for idx, (metric, question) in enumerate(questions_list, 1):
            print(f"\n[Question {idx}/10]")
            print(f"🔹 {question}")
            answer = input("Your answer: ")
            assistant.responses[metric] = answer
            assistant.scores[metric] = assistant.score_answer(answer)
            
            if idx % 3 == 0 and idx < 10:
                print("\n💬 You're doing great!")
        
        # PHQ-9
        print("\n" + "="*60)
        print("📋 PHQ-9 DEPRESSION SCREENING")
        print("="*60)
        print("Answer with: Never, Sometimes, or Often\n")
        
        phq9_responses = {}
        for idx, (category, question) in enumerate(assistant.phq9_questions.items(), 1):
            print(f"\n[PHQ-9 {idx}/9]")
            print(f"🔹 {question}")
            answer = input("Your answer: ")
            phq9_responses[category] = answer
        
        # Results
        phq9_score, severity, advice = assistant.calculate_phq9(phq9_responses)
        report, report_file = assistant.generate_report()
        
        print("\n" + "="*60)
        print("📊 RESULTS")
        print("="*60)
        print(f"\nAverage Score: {report['average_score']:.2f}")
        if report['high_risk_areas']:
            print(f"High Risk Areas: {', '.join(report['high_risk_areas'])}")
        print(f"\nPHQ-9 Score: {phq9_score}/27")
        print(f"Severity: {severity}")
        print(f"Advice: {advice}")
        print(f"\nOverall: {report['risk_level']}")
        print(f"\n💾 Report saved: {report_file}\n")


if __name__ == "__main__":
    main()

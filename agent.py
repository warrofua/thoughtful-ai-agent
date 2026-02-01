"""
Core Agent logic for Thoughtful AI Customer Support.
"""

import os
import sys
import warnings
import random
import numpy as np
from typing import Optional
from sentence_transformers import SentenceTransformer

from data import (
    QUESTIONS, ANSWERS,
    GREETING_RESPONSES, HELP_RESPONSES, FAREWELL_RESPONSES,
    GRATITUDE_RESPONSES, UNKNOWN_RESPONSES, ACKNOWLEDGMENT_RESPONSES,
    CONFUSION_RESPONSES, INTENT_KEYWORDS
)

# Suppress transformers/sentence-transformers warnings and logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")

# Suppress logging
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# Configuration
SIMILARITY_THRESHOLD = 0.55  # Threshold for matching predefined answers (balanced for short queries)
DEFAULT_MODEL = "all-MiniLM-L6-v2"  # Lightweight, fast embedding model

# Optional OpenAI import - gracefully handles if not installed
try:
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class ThoughtfulAIAgent:
    """
    Customer Support Agent for Thoughtful AI.
    
    Uses semantic search to match user queries against predefined Q&A.
    Falls back to context-appropriate generic responses for questions 
    outside the predefined dataset.
    
    Optional: Set OPENAI_API_KEY in .env for enhanced LLM fallback.
    """
    
    def __init__(self):
        self.predefined_embeddings = None
        self.openai_client = None
        self.openai_enabled = False
        
        # Track which responses we've used (for variety)
        self._response_counters = {
            "greeting": 0,
            "help": 0,
            "farewell": 0,
            "gratitude": 0,
            "unknown": 0,
            "acknowledgment": 0,
            "confusion": 0,
        }
        
        # Load model with suppressed output
        self.embedding_model = self._load_model_silently()
        
        # Pre-compute embeddings for predefined questions
        self._compute_embeddings()
        
        # Try to initialize OpenAI (optional, silent if fails)
        self._init_openai_silently()
    
    def _load_model_silently(self):
        """Load the sentence transformer model with suppressed output."""
        from io import StringIO
        
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        
        try:
            model = SentenceTransformer(DEFAULT_MODEL)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        return model
    
    def _compute_embeddings(self):
        """Pre-compute embeddings for all predefined questions."""
        self.predefined_embeddings = self.embedding_model.encode(QUESTIONS, show_progress_bar=False)
        print("✅ Agent ready!", file=sys.stderr)
    
    def _init_openai_silently(self):
        """Initialize OpenAI client silently if key is available."""
        if not OPENAI_AVAILABLE:
            return
        
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and api_key.strip() and api_key != "your_openai_api_key_here":
            try:
                self.openai_client = OpenAI(api_key=api_key)
                self.openai_enabled = True
            except Exception:
                # Silently fail - we'll use generic responses
                pass
    
    def _find_best_match(self, query: str) -> tuple[Optional[str], float]:
        """
        Find the best matching predefined question using cosine similarity.
        
        Returns:
            Tuple of (matched_question, similarity_score) or (None, 0.0) if no match
        """
        query_embedding = self.embedding_model.encode([query])
        
        similarities = np.dot(self.predefined_embeddings, query_embedding.T).flatten()
        
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        if best_score >= SIMILARITY_THRESHOLD:
            return QUESTIONS[best_idx], float(best_score)
        return None, float(best_score)
    
    def _detect_intent(self, query: str) -> str:
        """
        Detect the user's intent from their query.
        
        Returns:
            Intent category: 'greeting', 'help', 'farewell', 'gratitude', 
            'acknowledgment', 'confusion', or 'unknown'
        """
        query_lower = query.lower().strip()
        words = query_lower.split()
        words_set = set(words)
        
        # Priority 1: Help/capabilities (check these first since they're more specific)
        help_keywords = ["help", "what can you do", "what do you do", "capabilities", 
                        "what are you", "who are you", "features", "functions", "assist"]
        for keyword in help_keywords:
            if keyword in query_lower:
                return "help"
        
        # Priority 2: Check for exact word matches for greetings
        greeting_words = ["hi", "hello", "hey", "greetings", "howdy", "hiya", "yo", "sup"]
        if any(word in words_set for word in greeting_words):
            return "greeting"
        
        # Check for time-based greetings
        if any(word in query_lower for word in ["morning", "afternoon", "evening"]):
            return "greeting"
        
        # Priority 3: Farewell
        farewell_words = ["bye", "goodbye", "cya", "later"]
        if any(word in words_set for word in farewell_words):
            return "farewell"
        if "see you" in query_lower or "exit" in words_set or "quit" in words_set:
            return "farewell"
        
        # Priority 4: Gratitude
        gratitude_words = ["thanks", "thank", "thx", "ty", "appreciate", "grateful", "cheers"]
        if any(word in words_set for word in gratitude_words):
            return "gratitude"
        
        # Priority 5: Acknowledgment
        ack_words = ["ok", "okay", "cool", "great", "good", "nice", "perfect", "sure", "alright"]
        if any(word in words_set for word in ack_words):
            return "acknowledgment"
        
        # Priority 6: Confusion/unclear
        confusion_words = ["what", "huh", "confused"]
        if any(word in words_set for word in confusion_words[:2]):
            return "confusion"
        if "don't understand" in query_lower or "dont understand" in query_lower:
            return "confusion"
        
        return "unknown"
    
    def _get_generic_response(self, intent: str) -> tuple[str, str]:
        """
        Get a context-appropriate generic response.
        
        Returns:
            Tuple of (response_text, source_category)
        """
        response_map = {
            "greeting": (GREETING_RESPONSES, "generic-greeting"),
            "help": (HELP_RESPONSES, "generic-help"),
            "farewell": (FAREWELL_RESPONSES, "generic-farewell"),
            "gratitude": (GRATITUDE_RESPONSES, "generic-gratitude"),
            "acknowledgment": (ACKNOWLEDGMENT_RESPONSES, "generic-ack"),
            "confusion": (CONFUSION_RESPONSES, "generic-confusion"),
            "unknown": (UNKNOWN_RESPONSES, "generic-unknown"),
        }
        
        responses, source = response_map.get(intent, response_map["unknown"])
        
        # Cycle through responses for variety
        counter_key = intent if intent in self._response_counters else "unknown"
        index = self._response_counters[counter_key] % len(responses)
        self._response_counters[counter_key] += 1
        
        return responses[index], source
    
    def _get_openai_response(self, query: str) -> Optional[str]:
        """
        Get a response from OpenAI for unknown questions.
        Returns None if OpenAI is not available or fails.
        """
        if not self.openai_enabled or not self.openai_client:
            return None
        
        try:
            system_prompt = (
                "You are a helpful customer support agent for Thoughtful AI, a company that "
                "provides AI-powered automation agents for healthcare. "
                "Thoughtful AI offers agents like EVA (Eligibility Verification), "
                "CAM (Claims Processing), and PHIL (Payment Posting). "
                "The user asked something not directly in your training data. "
                "Provide a helpful, friendly response. If you can relate their question "
                "to healthcare automation, do so. Otherwise, gently guide them back to "
                "topics about Thoughtful AI's agents. Keep it brief (2-3 sentences)."
            )
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.7,
                max_tokens=150
            )
            
            return response.choices[0].message.content
        except Exception:
            # Fall back to generic responses on any error
            return None
    
    def respond(self, query: str) -> dict:
        """
        Generate a response to the user's query.
        
        Returns:
            Dictionary containing:
                - response: The answer text
                - source: 'predefined', 'generic-*', or 'llm'
                - confidence: Similarity score (for predefined) or None
        """
        query = query.strip()
        
        if not query:
            return {
                "response": "Please enter a valid question.",
                "source": "error",
                "confidence": None
            }
        
        # Try to find a match in predefined Q&A
        matched_question, score = self._find_best_match(query)
        
        if matched_question:
            return {
                "response": ANSWERS[matched_question],
                "source": "predefined",
                "confidence": score
            }
        
        # No predefined match - detect intent
        intent = self._detect_intent(query)
        
        # For unknown intent, try OpenAI first if available
        if intent == "unknown" and self.openai_enabled:
            openai_response = self._get_openai_response(query)
            if openai_response:
                return {
                    "response": openai_response,
                    "source": "llm",
                    "confidence": None
                }
        
        # Use generic response (either as fallback or primary for non-unknown intents)
        response, source = self._get_generic_response(intent)
        
        return {
            "response": response,
            "source": source,
            "confidence": None
        }

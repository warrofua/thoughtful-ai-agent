"""
Core Agent logic for Thoughtful AI Customer Support.

This module implements a conversational AI agent that answers questions about
Thoughtful AI's healthcare automation agents (EVA, CAM, PHIL) using a multi-layer
matching strategy:

1. Facet-based matching - Matches functional descriptions (e.g., "handle claims" → CAM)
2. Semantic search - Uses sentence embeddings for similarity matching
3. Intent detection - Recognizes greetings, help requests, thanks, etc.
4. Generic fallback - Context-appropriate responses for unknown queries

Optional: OpenAI integration for enhanced LLM-based fallback responses.

Example:
    >>> agent = ThoughtfulAIAgent()
    >>> result = agent.respond("What is EVA?")
    >>> print(result['response'])
    'EVA automates the process of verifying a patient's eligibility...'
"""

import os
import sys
import warnings
import random
import numpy as np
from typing import Optional, Tuple
from sentence_transformers import SentenceTransformer

# Import data structures containing predefined Q&A and generic responses
from data import (
    QUESTIONS, ANSWERS, FACET_MAP,
    GREETING_RESPONSES, HELP_RESPONSES, FAREWELL_RESPONSES,
    GRATITUDE_RESPONSES, UNKNOWN_RESPONSES, ACKNOWLEDGMENT_RESPONSES,
    CONFUSION_RESPONSES, INTENT_KEYWORDS
)

# ============================================================================
# CONFIGURATION & LOGGING SETUP
# ============================================================================

# Disable tokenizers parallelism warnings (not needed for inference)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Filter out common warnings from transformers and huggingface libraries
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")

# Set logging levels to ERROR only to suppress verbose output
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# Configuration constants
SIMILARITY_THRESHOLD = 0.55  # Minimum cosine similarity for semantic matches
DEFAULT_MODEL = "all-MiniLM-L6-v2"  # Lightweight embedding model (~80MB)

# Optional OpenAI integration - gracefully handles if not installed
try:
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class ThoughtfulAIAgent:
    """
    Customer Support Agent for Thoughtful AI.
    
    This agent uses a cascading matching strategy to answer user queries:
    
    Priority 1: Facet-based matching
        - Matches functional descriptions (e.g., "process claims") 
        - Doesn't require user to know agent names (CAM, EVA, PHIL)
        
    Priority 2: Semantic search
        - Uses sentence embeddings to find similar predefined questions
        - Handles typos, variations, and rephrased questions
        
    Priority 3: Intent detection
        - Recognizes greetings, help requests, thanks, farewells, etc.
        - Provides context-appropriate generic responses
        
    Priority 4: Generic fallback
        - Polite responses for unknown questions
        - Optional: OpenAI GPT for enhanced responses
    
    Attributes:
        embedding_model: SentenceTransformer model for semantic search
        predefined_embeddings: Pre-computed embeddings of all predefined questions
        openai_client: Optional OpenAI client for LLM fallback
        openai_enabled: Whether OpenAI integration is active
        _response_counters: Tracks which generic responses have been used
    
    Example:
        >>> agent = ThoughtfulAIAgent()
        >>> result = agent.respond("How do I verify eligibility?")
        >>> print(result['source'])  # 'predefined'
        >>> print(result['response'])  # EVA description
    """
    
    def __init__(self):
        """
        Initialize the agent.
        
        This performs several setup steps:
        1. Initializes the sentence transformer model (silent)
        2. Pre-computes embeddings for all predefined questions
        3. Attempts to initialize OpenAI client (optional, silent on failure)
        4. Sets up response rotation counters for variety
        """
        self.predefined_embeddings = None
        self.openai_client = None
        self.openai_enabled = False
        
        # Initialize counters for response rotation (ensures variety)
        self._response_counters = {
            "greeting": 0,
            "help": 0,
            "farewell": 0,
            "gratitude": 0,
            "unknown": 0,
            "acknowledgment": 0,
            "confusion": 0,
        }
        
        # Load the embedding model with suppressed output
        self.embedding_model = self._load_model_silently()
        
        # Pre-compute embeddings for all predefined questions
        # This is done once at initialization for fast query-time matching
        self._compute_embeddings()
        
        # Attempt to initialize OpenAI (optional, doesn't affect core functionality)
        self._init_openai_silently()
    
    def _load_model_silently(self) -> SentenceTransformer:
        """
        Load the sentence transformer model without printing to stdout.
        
        The sentence-transformers library prints verbose output during model
        loading. We suppress this to keep the UI clean.
        
        Returns:
            Loaded SentenceTransformer model
        """
        from io import StringIO
        
        # Capture and discard stdout/stderr during model loading
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        
        try:
            model = SentenceTransformer(DEFAULT_MODEL)
        finally:
            # Restore output streams
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        return model
    
    def _compute_embeddings(self) -> None:
        """
        Pre-compute embeddings for all predefined questions.
        
        This is an optimization that encodes all predefined questions once
        at startup. At query time, we only need to encode the user's query
        and compute cosine similarities.
        
        The embeddings are stored in self.predefined_embeddings.
        """
        self.predefined_embeddings = self.embedding_model.encode(
            QUESTIONS, 
            show_progress_bar=False  # Suppress tqdm progress bar
        )
        print("✅ Agent ready!", file=sys.stderr)
    
    def _init_openai_silently(self) -> None:
        """
        Initialize OpenAI client if API key is available.
        
        This method attempts to:
        1. Check if OpenAI package is installed
        2. Read API key from environment variable
        3. Initialize the client
        
        If any step fails, the agent gracefully falls back to generic responses.
        No error messages are shown to maintain clean UX.
        """
        if not OPENAI_AVAILABLE:
            return
        
        api_key = os.getenv("OPENAI_API_KEY")
        
        # Validate API key (not empty, not placeholder)
        if api_key and api_key.strip() and api_key != "your_openai_api_key_here":
            try:
                self.openai_client = OpenAI(api_key=api_key)
                self.openai_enabled = True
            except Exception:
                # Silently fail - generic responses work perfectly fine
                pass
    
    def _find_best_match(self, query: str) -> Tuple[Optional[str], float]:
        """
        Find the best matching predefined question using cosine similarity.
        
        This method encodes the user's query and compares it against all
        pre-computed question embeddings using cosine similarity.
        
        Args:
            query: The user's input query
            
        Returns:
            Tuple of (matched_question, similarity_score)
            - matched_question: The best matching predefined question, or None
            - similarity_score: Cosine similarity (0.0 to 1.0)
            
        Note:
            A match is only returned if similarity >= SIMILARITY_THRESHOLD
        """
        # Encode the user's query (this is the expensive operation)
        query_embedding = self.embedding_model.encode([query])
        
        # Compute cosine similarities with all predefined questions
        # np.dot computes the dot product; for normalized vectors, this equals cosine similarity
        similarities = np.dot(self.predefined_embeddings, query_embedding.T).flatten()
        
        # Find the best matching question
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        # Only return match if it meets the threshold
        if best_score >= SIMILARITY_THRESHOLD:
            return QUESTIONS[best_idx], float(best_score)
        
        return None, float(best_score)
    
    def _find_facet_match(self, query: str) -> Optional[str]:
        """
        Check if query matches any facet keywords (functional descriptions).
        
        Facet matching allows users to ask about functionality without knowing
        agent names. For example:
        - "how do you handle claims" → CAM answer
        - "how do you verify eligibility" → EVA answer
        - "how do you post payments" → PHIL answer
        
        This is Priority 1 matching because it's more reliable than semantic
        search for functional queries.
        
        Args:
            query: The user's input query
            
        Returns:
            The matching answer string, or None if no facet matches
            
        Algorithm:
            1. Check for exact substring matches in facet keywords
            2. If no exact match, check for word overlap (>= 2 words)
        """
        query_lower = query.lower()
        
        # Priority 1: Check for exact substring matches
        # This catches queries like "do you process claims"
        for facet, answer in FACET_MAP.items():
            if facet in query_lower:
                return answer
        
        # Priority 2: Check for word overlap
        # This catches variations like "claims processing" vs "process claims"
        query_words = set(query_lower.split())
        best_match = None
        best_score = 0
        
        for facet, answer in FACET_MAP.items():
            facet_words = set(facet.split())
            overlap = len(query_words & facet_words)
            
            # Require at least 2 words to match for better precision
            # This prevents false matches on single common words
            if overlap >= 2 and overlap > best_score:
                best_score = overlap
                best_match = answer
        
        return best_match
    
    def _detect_intent(self, query: str) -> str:
        """
        Detect the user's intent from their query.
        
        This method uses keyword matching to categorize user input into
        predefined intent categories. This allows the agent to provide
        context-appropriate responses.
        
        Args:
            query: The user's input query
            
        Returns:
            Intent category string:
            - 'greeting': User is saying hello
            - 'help': User wants to know capabilities
            - 'farewell': User is saying goodbye
            - 'gratitude': User is saying thanks
            - 'acknowledgment': User acknowledged something
            - 'confusion': User is confused/unclear
            - 'unknown': No specific intent detected
            
        Priority Order:
            1. Help (most specific, avoids matching "hi help" as greeting)
            2. Greeting
            3. Farewell
            4. Gratitude
            5. Acknowledgment
            6. Confusion
            7. Unknown (default)
        """
        query_lower = query.lower().strip()
        words = query_lower.split()
        words_set = set(words)
        
        # Priority 1: Help/capabilities
        # Check these first since they're more specific than greetings
        help_keywords = [
            "help", "what can you do", "what do you do", "capabilities",
            "what are you", "who are you", "features", "functions", "assist"
        ]
        for keyword in help_keywords:
            if keyword in query_lower:
                return "help"
        
        # Priority 2: Greetings
        greeting_words = ["hi", "hello", "hey", "greetings", "howdy", "hiya", "yo", "sup"]
        if any(word in words_set for word in greeting_words):
            return "greeting"
        
        # Check for time-based greetings ("good morning", etc.)
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
        
        # Default: Unknown intent
        return "unknown"
    
    def _get_generic_response(self, intent: str) -> Tuple[str, str]:
        """
        Get a context-appropriate generic response.
        
        This method cycles through available responses for the given intent
        to provide variety in the conversation.
        
        Args:
            intent: The detected intent category
            
        Returns:
            Tuple of (response_text, source_category)
            - response_text: The selected response string
            - source_category: The source label (e.g., 'generic-greeting')
            
        Example:
            >>> response, source = agent._get_generic_response("greeting")
            >>> print(response)
            'Hello! 👋 Welcome to Thoughtful AI Support...'
            >>> print(source)
            'generic-greeting'
        """
        # Map intents to their response lists and source labels
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
        # This ensures users don't see the same response every time
        counter_key = intent if intent in self._response_counters else "unknown"
        index = self._response_counters[counter_key] % len(responses)
        self._response_counters[counter_key] += 1
        
        return responses[index], source
    
    def _get_openai_response(self, query: str) -> Optional[str]:
        """
        Get a response from OpenAI for unknown questions.
        
        This is an optional enhancement that uses GPT-3.5 to generate
        contextual responses for questions outside the predefined dataset.
        
        Args:
            query: The user's input query
            
        Returns:
            Generated response string, or None if OpenAI is not available
            or the API call fails
            
        Note:
            This method is silent on failure - it returns None and lets
            the caller fall back to generic responses.
        """
        if not self.openai_enabled or not self.openai_client:
            return None
        
        try:
            # System prompt guides the model to stay on-brand
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
                temperature=0.7,  # Balance creativity and consistency
                max_tokens=150    # Keep responses concise
            )
            
            return response.choices[0].message.content
        except Exception:
            # Silently fall back to generic responses on any error
            # This ensures the agent always responds, even if OpenAI fails
            return None
    
    def respond(self, query: str) -> dict:
        """
        Generate a response to the user's query.
        
        This is the main entry point for the agent. It implements a cascading
        matching strategy to provide the best possible response.
        
        Matching Priority:
            1. Facet-based matching (functional descriptions)
            2. Semantic search (predefined questions)
            3. Intent detection (greetings, help, etc.)
            4. OpenAI fallback (if enabled)
            5. Generic fallback (context-appropriate)
        
        Args:
            query: The user's input query string
            
        Returns:
            Dictionary containing:
                - response (str): The answer text
                - source (str): Source category:
                    * 'predefined' - Matched from predefined Q&A
                    * 'generic-*' - Generic intent-based response
                    * 'llm' - Generated by OpenAI
                    * 'error' - Invalid input
                - confidence (float or None): Similarity score for predefined matches
                
        Example:
            >>> agent = ThoughtfulAIAgent()
            >>> result = agent.respond("What is EVA?")
            >>> print(result)
            {
                'response': 'EVA automates the process of verifying...',
                'source': 'predefined',
                'confidence': 0.96
            }
        """
        query = query.strip()
        
        # Handle empty input
        if not query:
            return {
                "response": "Please enter a valid question.",
                "source": "error",
                "confidence": None
            }
        
        # =========================================================================
        # PRIORITY 1: Facet-based matching
        # This catches functional queries without requiring agent names
        # Example: "how do you handle claims" → CAM answer
        # =========================================================================
        facet_answer = self._find_facet_match(query)
        if facet_answer:
            return {
                "response": facet_answer,
                "source": "predefined",
                "confidence": 0.85  # High confidence for facet matches
            }
        
        # =========================================================================
        # PRIORITY 2: Semantic search on predefined questions
        # Uses sentence embeddings to find similar questions
        # Example: "Tell me about CAM" → matches "What is CAM?"
        # =========================================================================
        matched_question, score = self._find_best_match(query)
        
        if matched_question:
            return {
                "response": ANSWERS[matched_question],
                "source": "predefined",
                "confidence": score
            }
        
        # =========================================================================
        # PRIORITY 3: Intent detection
        # Recognizes greetings, help requests, thanks, farewells, etc.
        # Example: "hi" → Greeting response
        # =========================================================================
        intent = self._detect_intent(query)
        
        # =========================================================================
        # PRIORITY 4: OpenAI fallback (if enabled)
        # Only for truly unknown queries, not for recognized intents
        # =========================================================================
        if intent == "unknown" and self.openai_enabled:
            openai_response = self._get_openai_response(query)
            if openai_response:
                return {
                    "response": openai_response,
                    "source": "llm",
                    "confidence": None
                }
        
        # =========================================================================
        # PRIORITY 5: Generic fallback response
        # Context-appropriate response based on detected intent
        # =========================================================================
        response, source = self._get_generic_response(intent)
        
        return {
            "response": response,
            "source": source,
            "confidence": None
        }

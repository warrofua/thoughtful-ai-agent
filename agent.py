"""
Core Agent logic for Thoughtful AI Customer Support.
"""

import os
import warnings
import numpy as np
from typing import Optional
from sentence_transformers import SentenceTransformer

from data import QUESTIONS, ANSWERS

# Suppress transformers/sentence-transformers warnings and logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")

# Suppress transformers logging
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# Configuration
SIMILARITY_THRESHOLD = 0.55  # Threshold for matching predefined answers (balanced for short queries)
DEFAULT_MODEL = "all-MiniLM-L6-v2"  # Lightweight, fast embedding model

# Generic fallback responses for questions outside the dataset
GENERIC_RESPONSES = [
    "I'm not sure about that. I can help you with questions about Thoughtful AI's agents like EVA, CAM, and PHIL. Is there something specific about our healthcare automation agents I can help you with?",
    "I don't have information on that topic. I'm specifically designed to answer questions about Thoughtful AI's agents (EVA, CAM, PHIL) and their benefits. How can I help you with those?",
    "That's outside my area of expertise. I specialize in Thoughtful AI's healthcare automation agents. Would you like to know about EVA, CAM, or PHIL instead?",
]


class ThoughtfulAIAgent:
    """
    Customer Support Agent for Thoughtful AI.
    
    Uses semantic search to match user queries against predefined Q&A.
    Falls back to generic responses for questions outside the predefined dataset.
    """
    
    def __init__(self):
        self.predefined_embeddings = None
        self._fallback_index = 0
        
        # Load model with suppressed output
        self.embedding_model = self._load_model_silently()
        
        # Pre-compute embeddings for predefined questions
        self._compute_embeddings()
    
    def _load_model_silently(self):
        """Load the sentence transformer model with suppressed output."""
        import sys
        from io import StringIO
        
        # Capture stdout/stderr during model loading
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
        import sys
        # Suppress stdout during model loading
        self.predefined_embeddings = self.embedding_model.encode(QUESTIONS, show_progress_bar=False)
        print("✅ Agent ready!", file=sys.stderr)
    
    def _find_best_match(self, query: str) -> tuple[Optional[str], float]:
        """
        Find the best matching predefined question using cosine similarity.
        
        Returns:
            Tuple of (matched_question, similarity_score) or (None, 0.0) if no match
        """
        query_embedding = self.embedding_model.encode([query])
        
        # Calculate cosine similarities
        similarities = np.dot(self.predefined_embeddings, query_embedding.T).flatten()
        
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        if best_score >= SIMILARITY_THRESHOLD:
            return QUESTIONS[best_idx], float(best_score)
        return None, float(best_score)
    
    def _get_generic_response(self, query: str) -> str:
        """
        Get a generic fallback response for questions not in the predefined dataset.
        Cycles through different responses for variety.
        """
        response = GENERIC_RESPONSES[self._fallback_index % len(GENERIC_RESPONSES)]
        self._fallback_index += 1
        return response
    
    def respond(self, query: str) -> dict:
        """
        Generate a response to the user's query.
        
        Returns:
            Dictionary containing:
                - response: The answer text
                - source: 'predefined' or 'generic'
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
        
        # Fall back to generic response
        return {
            "response": self._get_generic_response(query),
            "source": "generic",
            "confidence": None
        }

"""
Core Agent logic for Thoughtful AI Customer Support.
"""

import os
import numpy as np
from typing import Optional
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

from data import QUESTIONS, ANSWERS

# Load environment variables
load_dotenv()

# Configuration
SIMILARITY_THRESHOLD = 0.55  # Threshold for matching predefined answers (balanced for short queries)
DEFAULT_MODEL = "all-MiniLM-L6-v2"  # Lightweight, fast embedding model


class ThoughtfulAIAgent:
    """
    Customer Support Agent for Thoughtful AI.
    
    Uses semantic search to match user queries against predefined Q&A.
    Falls back to LLM for questions outside the predefined dataset.
    """
    
    def __init__(self):
        self.embedding_model = SentenceTransformer(DEFAULT_MODEL)
        self.predefined_embeddings = None
        self.openai_client = None
        
        # Pre-compute embeddings for predefined questions
        self._compute_embeddings()
        
        # Initialize OpenAI client if API key is available
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.openai_client = OpenAI(api_key=api_key)
    
    def _compute_embeddings(self):
        """Pre-compute embeddings for all predefined questions."""
        import sys
        print("🔄 Loading embedding model...", file=sys.stderr)
        self.predefined_embeddings = self.embedding_model.encode(QUESTIONS)
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
    
    def _get_llm_response(self, query: str) -> str:
        """
        Get a response from the LLM for questions not in the predefined dataset.
        """
        if not self.openai_client:
            return (
                "I'm not sure about that. I can help you with questions about Thoughtful AI's "
                "agents like EVA, CAM, and PHIL. For other questions, please set up an OpenAI API key."
            )
        
        try:
            system_prompt = (
                "You are a helpful customer support agent for Thoughtful AI, a company that "
                "provides AI-powered automation agents for healthcare. "
                "Thoughtful AI offers agents like EVA (Eligibility Verification), "
                "CAM (Claims Processing), and PHIL (Payment Posting). "
                "Answer the user's question helpfully. If you don't know something specific "
                "about Thoughtful AI, provide a general helpful response."
            )
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    
    def respond(self, query: str) -> dict:
        """
        Generate a response to the user's query.
        
        Returns:
            Dictionary containing:
                - response: The answer text
                - source: 'predefined' or 'llm'
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
        
        # Fall back to LLM
        return {
            "response": self._get_llm_response(query),
            "source": "llm",
            "confidence": None
        }

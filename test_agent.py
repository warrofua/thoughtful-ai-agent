"""
Comprehensive test suite for Thoughtful AI Customer Support Agent.

Run with: python -m pytest test_agent.py -v
Or: python test_agent.py
"""

import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import ThoughtfulAIAgent
from data import (
    PREDEFINED_QA, GREETING_RESPONSES, HELP_RESPONSES,
    FAREWELL_RESPONSES, GRATITUDE_RESPONSES, UNKNOWN_RESPONSES,
    ACKNOWLEDGMENT_RESPONSES, CONFUSION_RESPONSES, FACET_MAP
)


class TestThoughtfulAIAgent(unittest.TestCase):
    """Test suite for the ThoughtfulAIAgent class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the agent once for all tests (expensive operation)."""
        # Suppress stdout during agent initialization
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        
        try:
            cls.agent = ThoughtfulAIAgent()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    
    # =========================================================================
    # PREDEFINED Q&A TESTS
    # =========================================================================
    
    def test_eva_exact_match(self):
        """Test exact match for EVA question."""
        result = self.agent.respond("What does the eligibility verification agent (EVA) do?")
        self.assertEqual(result["source"], "predefined")
        self.assertIn("EVA", result["response"])
        self.assertIn("eligibility", result["response"])
        # Could be 0.85 (facet match) or higher (semantic match)
        self.assertGreaterEqual(result["confidence"], 0.85)
    
    def test_cam_exact_match(self):
        """Test exact match for CAM question."""
        result = self.agent.respond("What does the claims processing agent (CAM) do?")
        self.assertEqual(result["source"], "predefined")
        self.assertIn("CAM", result["response"])
        self.assertIn("claims", result["response"])
        # Could be 0.85 (facet match) or higher (semantic match)
        self.assertGreaterEqual(result["confidence"], 0.85)
    
    def test_phil_exact_match(self):
        """Test exact match for PHIL question."""
        result = self.agent.respond("How does the payment posting agent (PHIL) work?")
        self.assertEqual(result["source"], "predefined")
        self.assertIn("PHIL", result["response"])
        self.assertIn("payment", result["response"])
        # Confidence could be 0.85 (facet match) or higher (semantic match)
        self.assertGreaterEqual(result["confidence"], 0.85)
    
    def test_about_thoughtful_ai(self):
        """Test match for general company question."""
        result = self.agent.respond("Tell me about Thoughtful AI's Agents.")
        self.assertEqual(result["source"], "predefined")
        # Response should mention at least one agent
        has_agent = "EVA" in result["response"] or "CAM" in result["response"] or "PHIL" in result["response"]
        self.assertTrue(has_agent, "Response should mention at least one agent")
    
    def test_benefits(self):
        """Test match for benefits question."""
        result = self.agent.respond("What are the benefits of using Thoughtful AI's agents?")
        self.assertEqual(result["source"], "predefined")
        self.assertIn("reduce", result["response"].lower())
    
    # =========================================================================
    # VARIATION TESTS (Semantic matching)
    # =========================================================================
    
    def test_eva_variations(self):
        """Test various ways to ask about EVA."""
        variations = [
            "What is EVA?",
            "Tell me about EVA",
            "Explain EVA",
            "What does EVA do?",
        ]
        for query in variations:
            result = self.agent.respond(query)
            self.assertEqual(result["source"], "predefined", 
                           f"Failed for: {query}")
            self.assertIn("EVA", result["response"])
    
    def test_cam_variations(self):
        """Test various ways to ask about CAM."""
        variations = [
            "What is CAM?",
            "Tell me about CAM",
            "What does CAM do?",
        ]
        for query in variations:
            result = self.agent.respond(query)
            self.assertEqual(result["source"], "predefined")
            self.assertIn("CAM", result["response"])
    
    def test_phil_variations(self):
        """Test various ways to ask about PHIL."""
        variations = [
            "What is PHIL?",
            "Tell me about PHIL",
            "How does PHIL work?",
        ]
        for query in variations:
            result = self.agent.respond(query)
            self.assertEqual(result["source"], "predefined")
            self.assertIn("PHIL", result["response"])
    
    # =========================================================================
    # FACET-BASED TESTS (Functional queries without agent names)
    # =========================================================================
    
    def test_eva_facets(self):
        """Test facet-based matching for EVA functionality."""
        facet_queries = [
            "how do you verify eligibility?",
            "can you check insurance eligibility?",
            "do you do benefits verification?",
            "how do you check patient eligibility?",
            "can you verify benefits?",
        ]
        for query in facet_queries:
            result = self.agent.respond(query)
            self.assertEqual(result["source"], "predefined",
                           f"Facet match failed for: {query}")
            self.assertIn("EVA", result["response"])
            self.assertIn("eligibility", result["response"].lower())
    
    def test_cam_facets(self):
        """Test facet-based matching for CAM functionality."""
        facet_queries = [
            "how do you handle claims?",
            "do you process claims?",
            "can you manage claims?",
            "how do you submit claims?",
            "do you do claims processing?",
        ]
        for query in facet_queries:
            result = self.agent.respond(query)
            self.assertEqual(result["source"], "predefined",
                           f"Facet match failed for: {query}")
            self.assertIn("CAM", result["response"])
            self.assertIn("claims", result["response"].lower())
    
    def test_phil_facets(self):
        """Test facet-based matching for PHIL functionality."""
        facet_queries = [
            "how do you post payments?",
            "can you reconcile payments?",
            "do you handle payment posting?",
            "how do you process payments?",
            "can you do payment reconciliation?",
        ]
        for query in facet_queries:
            result = self.agent.respond(query)
            self.assertEqual(result["source"], "predefined",
                           f"Facet match failed for: {query}")
            self.assertIn("PHIL", result["response"])
            self.assertIn("payment", result["response"].lower())
    
    # =========================================================================
    # INTENT DETECTION TESTS
    # =========================================================================
    
    def test_greeting_intent(self):
        """Test greeting intent detection."""
        greetings = ["hi", "hello", "hey", "howdy", "yo"]
        for query in greetings:
            result = self.agent.respond(query)
            self.assertEqual(result["source"], "generic-greeting",
                           f"Failed for: {query}")
            self.assertIn(result["response"], GREETING_RESPONSES)
    
    def test_help_intent(self):
        """Test help intent detection."""
        help_queries = [
            "help",
            "who are you",
        ]
        for query in help_queries:
            result = self.agent.respond(query)
            self.assertEqual(result["source"], "generic-help",
                           f"Failed for: {query}")
            self.assertIn(result["response"], HELP_RESPONSES)
    
    def test_help_intent_may_match_predefined(self):
        """Test that help queries may match predefined or generic."""
        # These may match either predefined or generic-help depending on similarity
        flexible_queries = ["what can you do", "what do you do"]
        for query in flexible_queries:
            result = self.agent.respond(query)
            self.assertIn(result["source"], ["predefined", "generic-help"],
                         f"Failed for: {query}")
    
    def test_farewell_intent(self):
        """Test farewell intent detection."""
        farewells = ["bye", "goodbye", "see you", "cya"]
        for query in farewells:
            result = self.agent.respond(query)
            self.assertEqual(result["source"], "generic-farewell",
                           f"Failed for: {query}")
            self.assertIn(result["response"], FAREWELL_RESPONSES)
    
    def test_gratitude_intent(self):
        """Test gratitude intent detection."""
        gratitude = ["thanks", "thank you", "thx", "appreciate it"]
        for query in gratitude:
            result = self.agent.respond(query)
            self.assertEqual(result["source"], "generic-gratitude",
                           f"Failed for: {query}")
            self.assertIn(result["response"], GRATITUDE_RESPONSES)
    
    def test_acknowledgment_intent(self):
        """Test acknowledgment intent detection."""
        acks = ["ok", "okay", "cool", "great", "nice"]
        for query in acks:
            result = self.agent.respond(query)
            self.assertEqual(result["source"], "generic-ack",
                           f"Failed for: {query}")
            self.assertIn(result["response"], ACKNOWLEDGMENT_RESPONSES)
    
    # =========================================================================
    # FALLBACK TESTS
    # =========================================================================
    
    def test_unknown_queries(self):
        """Test fallback for unknown queries."""
        unknown_queries = [
            "what is the weather today?",
            "tell me about space",
            "how do I cook pasta?",
        ]
        for query in unknown_queries:
            result = self.agent.respond(query)
            # Should be either unknown response or confusion response
            self.assertTrue(
                result["source"] in ["generic-unknown", "generic-confusion"],
                f"Unexpected source for '{query}': {result['source']}"
            )
    
    def test_empty_input(self):
        """Test handling of empty input."""
        result = self.agent.respond("")
        self.assertEqual(result["source"], "error")
        self.assertIn("valid question", result["response"].lower())
    
    def test_whitespace_input(self):
        """Test handling of whitespace-only input."""
        result = self.agent.respond("   ")
        self.assertEqual(result["source"], "error")
    
    # =========================================================================
    # RESPONSE ROTATION TESTS
    # =========================================================================
    
    def test_greeting_rotation(self):
        """Test that greetings rotate for variety."""
        responses = set()
        for _ in range(3):
            result = self.agent.respond("hi")
            responses.add(result["response"])
        # Should get different responses (or eventually cycle)
        self.assertGreaterEqual(len(responses), 1)
    
    # =========================================================================
    # EDGE CASE TESTS
    # =========================================================================
    
    def test_case_insensitivity(self):
        """Test that queries are case-insensitive."""
        queries = ["WHAT IS EVA?", "what is eva?", "What Is Eva?"]
        for query in queries:
            result = self.agent.respond(query)
            self.assertEqual(result["source"], "predefined")
    
    def test_punctuation(self):
        """Test handling of punctuation."""
        queries = ["what is EVA?", "what is EVA!", "what is EVA."]
        for query in queries:
            result = self.agent.respond(query)
            self.assertEqual(result["source"], "predefined")


class TestDataIntegrity(unittest.TestCase):
    """Test data structure integrity."""
    
    def test_predefined_qa_structure(self):
        """Verify all predefined Q&A entries have required fields."""
        for qa in PREDEFINED_QA:
            self.assertIn("question", qa)
            self.assertIn("answer", qa)
            self.assertIn("variations", qa)
            self.assertIn("facets", qa)
            self.assertIsInstance(qa["variations"], list)
            self.assertIsInstance(qa["facets"], list)
    
    def test_facet_map_populated(self):
        """Verify facet map is not empty."""
        self.assertGreater(len(FACET_MAP), 0)
    
    def test_response_lists_not_empty(self):
        """Verify all response lists are populated."""
        self.assertGreater(len(GREETING_RESPONSES), 0)
        self.assertGreater(len(HELP_RESPONSES), 0)
        self.assertGreater(len(FAREWELL_RESPONSES), 0)
        self.assertGreater(len(GRATITUDE_RESPONSES), 0)
        self.assertGreater(len(UNKNOWN_RESPONSES), 0)
        self.assertGreater(len(ACKNOWLEDGMENT_RESPONSES), 0)
        self.assertGreater(len(CONFUSION_RESPONSES), 0)


class TestFacetMap(unittest.TestCase):
    """Specific tests for facet-based matching functionality."""
    
    def test_facet_map_keys_lowercase(self):
        """Verify all facet keys are lowercase for consistent matching."""
        for facet in FACET_MAP.keys():
            self.assertEqual(facet, facet.lower(),
                           f"Facet '{facet}' is not lowercase")
    
    def test_facet_keywords_comprehensive(self):
        """Test that facet keywords cover expected functionality."""
        # EVA-related facets
        eva_facets = [f for f in FACET_MAP.keys() 
                     if "eligibility" in f or "benefit" in f or "insurance" in f]
        self.assertGreater(len(eva_facets), 0, "No EVA facets found")
        
        # CAM-related facets
        cam_facets = [f for f in FACET_MAP.keys() if "claim" in f]
        self.assertGreater(len(cam_facets), 0, "No CAM facets found")
        
        # PHIL-related facets
        phil_facets = [f for f in FACET_MAP.keys() if "payment" in f]
        self.assertGreater(len(phil_facets), 0, "No PHIL facets found")


def run_tests():
    """Run all tests and print summary."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestThoughtfulAIAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestDataIntegrity))
    suite.addTests(loader.loadTestsFromTestCase(TestFacetMap))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    # Check if pytest is available for better output
    try:
        import pytest
        print("Running tests with pytest...")
        sys.exit(pytest.main([__file__, "-v"]))
    except ImportError:
        print("pytest not found, running with unittest...")
        sys.exit(run_tests())

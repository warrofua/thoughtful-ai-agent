"""
Test OpenAI integration for Thoughtful AI Customer Support Agent.

These tests verify that the OpenAI fallback works correctly when an API key is provided.
"""

import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class MockOpenAIResponse:
    """Mock OpenAI API response object."""
    def __init__(self, content):
        self.choices = [Mock(message=Mock(content=content))]


class TestOpenAIIntegration(unittest.TestCase):
    """Test suite for OpenAI integration."""
    
    def setUp(self):
        """Set up test environment."""
        # Suppress stdout during tests
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
    
    def tearDown(self):
        """Restore stdout/stderr."""
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-api-key-12345'})
    @patch('agent.OpenAI')
    def test_openai_initialization_with_valid_key(self, mock_openai_class):
        """Test that agent initializes OpenAI when valid API key is present."""
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Import and create agent after patching
        from agent import ThoughtfulAIAgent
        agent = ThoughtfulAIAgent()
        
        # Verify OpenAI was initialized
        self.assertTrue(agent.openai_enabled)
        self.assertIsNotNone(agent.openai_client)
        mock_openai_class.assert_called_once_with(api_key='test-api-key-12345')
    
    @patch.dict(os.environ, {}, clear=True)
    def test_openai_disabled_without_key(self):
        """Test that OpenAI is disabled when no API key is set."""
        from agent import ThoughtfulAIAgent
        
        # Clear any existing API key
        if 'OPENAI_API_KEY' in os.environ:
            del os.environ['OPENAI_API_KEY']
        
        agent = ThoughtfulAIAgent()
        
        # Verify OpenAI is not enabled
        self.assertFalse(agent.openai_enabled)
        self.assertIsNone(agent.openai_client)
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'your_openai_api_key_here'})
    def test_openai_disabled_with_placeholder_key(self):
        """Test that OpenAI is disabled when placeholder key is used."""
        from agent import ThoughtfulAIAgent
        
        agent = ThoughtfulAIAgent()
        
        # Verify OpenAI is not enabled with placeholder
        self.assertFalse(agent.openai_enabled)
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('agent.OpenAI')
    def test_openai_response_generation(self, mock_openai_class):
        """Test that OpenAI generates response for unknown queries."""
        # Setup mock client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Setup mock response
        mock_response = MockOpenAIResponse(
            "I'm not sure about that specific question, but I can help you with "
            "questions about EVA, CAM, and PHIL, our healthcare automation agents!"
        )
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create agent
        from agent import ThoughtfulAIAgent
        agent = ThoughtfulAIAgent()
        
        # Verify OpenAI is enabled
        self.assertTrue(agent.openai_enabled)
        
        # Get response for unknown query
        result = agent._get_openai_response("what's the weather like?")
        
        # Verify response was generated
        self.assertIsNotNone(result)
        self.assertIn("EVA", result)
        
        # Verify API was called correctly
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        
        # Check model
        self.assertEqual(call_args.kwargs['model'], 'gpt-3.5-turbo')
        
        # Check temperature and max_tokens
        self.assertEqual(call_args.kwargs['temperature'], 0.7)
        self.assertEqual(call_args.kwargs['max_tokens'], 150)
        
        # Check messages structure
        messages = call_args.kwargs['messages']
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]['role'], 'system')
        self.assertEqual(messages[1]['role'], 'user')
        self.assertEqual(messages[1]['content'], "what's the weather like?")
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('agent.OpenAI')
    def test_openai_graceful_failure(self, mock_openai_class):
        """Test that agent falls back to generic response on OpenAI error."""
        # Setup mock client to raise exception
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai_class.return_value = mock_client
        
        # Create agent
        from agent import ThoughtfulAIAgent
        agent = ThoughtfulAIAgent()
        
        # Verify OpenAI is enabled
        self.assertTrue(agent.openai_enabled)
        
        # Get response (should return None on error)
        result = agent._get_openai_response("some query")
        
        # Verify graceful failure
        self.assertIsNone(result)
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('agent.OpenAI')
    def test_openai_only_for_unknown_intent(self, mock_openai_class):
        """Test that OpenAI is only used for unknown intent, not greetings/etc."""
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Create agent
        from agent import ThoughtfulAIAgent
        agent = ThoughtfulAIAgent()
        
        # Greeting should not use OpenAI
        result = agent.respond("hi")
        self.assertEqual(result['source'], 'generic-greeting')
        mock_client.chat.completions.create.assert_not_called()
        
        # Reset mock
        mock_client.reset_mock()
        
        # Unknown query should use OpenAI
        mock_response = MockOpenAIResponse("I'm not sure about that.")
        mock_client.chat.completions.create.return_value = mock_response
        
        # Use a query that won't match any intent (not confusion/unknown trigger words)
        result = agent.respond("tell me about the solar system")
        self.assertEqual(result['source'], 'llm')
        mock_client.chat.completions.create.assert_called_once()


def run_tests():
    """Run OpenAI integration tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestOpenAIIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*70)
    print("OPENAI INTEGRATION TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ All OpenAI integration tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())

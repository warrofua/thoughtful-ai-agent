# 🤖 Thoughtful AI Customer Support Agent

A conversational AI Agent that answers questions about Thoughtful AI's healthcare automation agents using predefined responses, with intelligent generic fallback for unknown questions.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Rich](https://img.shields.io/badge/rich-terminal%20ui-green.svg)

## ✨ Features

- 🎯 **Semantic Search** - Matches user queries using sentence embeddings
- 💬 **Smart Intent Detection** - Recognizes greetings, help, thanks, farewells, etc.
- 🎨 **Beautiful Terminal UI** - Rich-powered interface with animations
- 🎬 **Startup Animation** - Professional loading sequence
- 📝 **50+ Generic Responses** - Context-appropriate fallbacks across 7 categories
- 🤖 **Optional OpenAI** - Enhanced responses when API key provided
- 📊 **Conversation Summary** - Review your chat on exit
- ⚡ **Zero Config** - Works immediately, no API keys required

## 🚀 Quick Start

```bash
git clone https://github.com/warrofua/thoughtful-ai-agent.git
cd thoughtful-ai-agent

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python main.py
```

## 🎮 Usage

### Commands

| Command | Description |
|---------|-------------|
| `hi`, `hello` | Get a greeting response |
| `what can you do` | See agent capabilities |
| `/examples` | Show example questions |
| `/help` | Display welcome message |
| `/quit`, `/exit` | Exit with conversation summary |

### Example Conversation

```
                        ╭─────────────────────────────╮
                        │   💙  Thoughtful AI  💙    │
                        │   Customer Support Agent    │
                        ╰─────────────────────────────╯

                           → Loading configuration...
                       → Connecting to knowledge base...
                      → Initializing semantic search...

╭──────────────────────────── Thoughtful AI Agent ─────────────────────────────╮
│                 👋 Welcome to Thoughtful AI Support!                         │
│                                                                              │
│                 I can help you with questions about:                         │
│                   • EVA (Eligibility Verification Agent)                     │
│                   • CAM (Claims Processing Agent)                            │
│                   • PHIL (Payment Posting Agent)                             │
│                   • General questions about Thoughtful AI                    │
╰──────────────────────────────────────────────────────────────────────────────╯

✅ Agent ready!
                                   🟢 Online

You: hi
╭──────────────────────────────────── You ─────────────────────────────────────╮
│ hi                                                                           │
╰──────────────────────────────────────────────────────────────────────────────╯
🤔 Thinking...
╭──────────────────────────── Thoughtful AI Agent ─────────────────────────────╮
│ Hello! 👋 Welcome to Thoughtful AI Support. I'm here to help you with        │
│ questions about our healthcare automation agents like EVA, CAM, and PHIL.    │
│ What would you like to know?                                                 │
╰──────────────────────────── 👋 Greeting response ────────────────────────────╯

You: what is EVA?
╭──────────────────────────────────── You ─────────────────────────────────────╮
│ what is EVA?                                                                 │
╰──────────────────────────────────────────────────────────────────────────────╯
🤔 Thinking...
╭──────────────────────────── Thoughtful AI Agent ─────────────────────────────╮
│ EVA automates the process of verifying a patient's eligibility and benefits  │
│ information in real-time, eliminating manual data entry errors and reducing  │
│ claim rejections.                                                            │
╰──────────────────── ✓ Predefined answer (confidence: 1.00) ──────────────────╯

You: /quit
───────────────────────────── Conversation Summary ─────────────────────────────
• Q: hi...
• A: Hello! 👋 Welcome to Thoughtful AI Support...
• Q: what is EVA?...
• A: EVA automates the process of verifying...

🌙 Closing session...
╭──────────────────────────────────────────────────────────────────────────────╮
│ Thank you for using Thoughtful AI Support. Goodbye! 👋                       │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## 🎯 How It Works

1. **Intent Detection** - Analyzes input for greetings, help requests, thanks, etc.
2. **Semantic Search** - Embeds query using `sentence-transformers`
3. **Similarity Matching** - Cosine similarity against predefined questions
4. **Smart Fallback** - Context-appropriate generic responses
5. **Optional Enhancement** - OpenAI GPT for smarter unknown responses

## 📋 Response Categories

| Category | Count | Triggers |
|----------|-------|----------|
| **Predefined** | 5 topics | EVA, CAM, PHIL, About, Benefits |
| **Greeting** | 8 | hi, hello, hey, yo |
| **Help** | 6 | help, what can you do |
| **Farewell** | 8 | bye, goodbye, see you |
| **Gratitude** | 8 | thanks, thank you |
| **Acknowledgment** | 8 | ok, cool, great |
| **Confusion** | 6 | what, huh, don't understand |
| **Unknown** | 10 | Anything else |

**Total: 59 different responses!**

## 🔧 Configuration (Optional)

For enhanced AI-generated fallback responses:

```bash
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=sk-...
```

Without an API key, the agent works perfectly with built-in responses.

## 📁 Project Structure

```
thoughtful-ai-agent/
├── agent.py          # Core logic with detailed documentation
├── data.py           # Q&A dataset + 54 generic responses
├── main.py           # Rich CLI with animations
├── test_agent.py     # Comprehensive test suite (28 tests)
├── requirements.txt  # Dependencies
├── .env.example      # Optional OpenAI config
└── README.md         # This file
```

## 📚 Code Documentation

All modules include comprehensive documentation:

- **Module docstrings** - Overview and usage examples
- **Class docstrings** - Attributes and methods
- **Method docstrings** - Args, Returns, Examples
- **Inline comments** - Explanation of complex logic
- **Type hints** - Function signatures with types

### Key Design Decisions (Documented in Code)

1. **Cascading Match Strategy** - Documented priority order in `respond()`
2. **Facet Matching** - Explains functional keyword detection
3. **Response Rotation** - Ensures variety in generic responses
4. **Silent Failures** - OpenAI optional, no errors if unavailable
5. **Confidence Color-Coding** - Visual feedback on match quality

## 🧪 Testing

### Quick Test

```bash
source venv/bin/activate
python3 -c "
from agent import ThoughtfulAIAgent
agent = ThoughtfulAIAgent()

for q in ['What is EVA?', 'hi', 'what can you do', 'thanks', 'bye']:
    r = agent.respond(q)
    print(f'{r[\"source\"]}: {q}')
"
```

### Comprehensive Test Suite

Run the full test suite (34 tests total):

```bash
source venv/bin/activate

# Main test suite (28 tests)
python test_agent.py

# OpenAI integration tests (6 tests)
python test_openai_integration.py
```

Or with pytest (if installed):

```bash
source venv/bin/activate
pip install pytest
pytest test_agent.py test_openai_integration.py -v
```

### Test Coverage

| Category | Tests | File |
|----------|-------|------|
| **Predefined Q&A** | Exact matches for EVA, CAM, PHIL, About, Benefits | `test_agent.py` |
| **Variations** | Semantic matching for different phrasings | `test_agent.py` |
| **Facet-based** | Functional queries without agent names | `test_agent.py` |
| **Intent Detection** | Greeting, help, farewell, gratitude, acknowledgment | `test_agent.py` |
| **Fallbacks** | Unknown queries, empty input, whitespace | `test_agent.py` |
| **Response Rotation** | Variety in generic responses | `test_agent.py` |
| **Edge Cases** | Case insensitivity, punctuation handling | `test_agent.py` |
| **Data Integrity** | Q&A structure, facet map, response lists | `test_agent.py` |
| **OpenAI Init** | API key validation, placeholder detection | `test_openai_integration.py` |
| **OpenAI Responses** | Response generation, error handling | `test_openai_integration.py` |
| **OpenAI Fallback** | Graceful degradation on API errors | `test_openai_integration.py` |

## 🏥 About Thoughtful AI

The agent knows about these healthcare automation agents:

| Agent | Full Name | Purpose |
|-------|-----------|---------|
| **EVA** | Eligibility Verification | Real-time benefits verification |
| **CAM** | Claims Processing | Streamlined claims submission |
| **PHIL** | Payment Posting | Automated payment reconciliation |

## 📦 Tech Stack

- **Python 3.10+**
- **Rich** - Terminal UI & animations
- **sentence-transformers** - Semantic embeddings
- **NumPy** - Vector operations
- **OpenAI** *(optional)* - LLM enhancement

## 📄 License

MIT License - Built for Thoughtful AI coding challenge.

---

<p align="center">
  <sub>Built with 💙 using <a href="https://github.com/Textualize/rich">Rich</a></sub>
</p>

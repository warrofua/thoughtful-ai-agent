# 🤖 Thoughtful AI Customer Support Agent

A conversational AI Agent that answers questions about Thoughtful AI's healthcare automation agents using predefined responses, with intelligent generic fallback for unknown questions.

## Features

- ✅ **Semantic Search**: Matches user queries to predefined Q&A using sentence embeddings
- ✅ **Conversational CLI**: Beautiful terminal interface with Rich
- ✅ **Smart Intent Detection**: Recognizes greetings, help requests, thanks, farewells, etc.
- ✅ **Variety of Responses**: 50+ generic responses across 7 intent categories
- ✅ **Optional OpenAI**: Enable LLM fallback for even smarter responses (optional, no key needed)
- ✅ **Smart Matching**: Handles typos, variations, and rephrased questions
- ✅ **Error Handling**: Graceful handling of invalid inputs

## Quick Start

### 1. Clone and Navigate

```bash
git clone https://github.com/warrofua/thoughtful-ai-agent.git
cd thoughtful-ai-agent
```

### 2. Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. (Optional) Enable OpenAI

For even smarter fallback responses, add your OpenAI API key:

```bash
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=your_key_here
```

Without an API key, the agent uses its built-in generic responses (works great!).

### 4. Run the Agent

```bash
python main.py
```

## Usage

```
👋 Welcome to Thoughtful AI Support!

I can help you with questions about:
  • EVA (Eligibility Verification Agent)
  • CAM (Claims Processing Agent)
  • PHIL (Payment Posting Agent)
  • General questions about Thoughtful AI

Commands:
  • Type your question and press Enter
  • Type /quit or /exit to exit
  • Type /help to see this message again
```

### Example Conversations

**Predefined Question:**
```
You: What does EVA do?

Thoughtful AI Agent: EVA automates the process of verifying a patient's 
eligibility and benefits information in real-time, eliminating manual data 
entry errors and reducing claim rejections.

✓ Predefined answer (confidence: 1.00)
```

**Greeting:**
```
You: hi

Thoughtful AI Agent: Hello! 👋 Welcome to Thoughtful AI Support. I'm here 
to help you with questions about our healthcare automation agents like EVA, 
CAM, and PHIL. What would you like to know?

👋 Greeting response
```

**Help Request:**
```
You: what can you do?

Thoughtful AI Agent: I can help you with questions about Thoughtful AI's 
healthcare automation agents! Specifically:

• **EVA** - Eligibility Verification Agent
• **CAM** - Claims Processing Agent  
• **PHIL** - Payment Posting Agent

What would you like to know about?

❓ Help response
```

**Generic Fallback:**
```
You: What's the weather like?

Thoughtful AI Agent: I'm not sure about that. I specialize in Thoughtful AI's 
healthcare automation agents like EVA, CAM, and PHIL. Is there something 
about those I can help you with?

💬 Unknown response
```

## How It Works

1. **Intent Detection**: First, the agent checks if your input matches common intents (greeting, help, thanks, etc.)
2. **Semantic Search**: If no intent match, your query is embedded using `sentence-transformers`
3. **Similarity Matching**: Cosine similarity compares query against predefined questions
4. **Threshold Check**: If similarity ≥ 0.55, return predefined answer
5. **Generic Fallback**: Otherwise, return a context-appropriate generic response
6. **Optional OpenAI**: If enabled, uses GPT for even smarter unknown question handling

## Response Categories

The agent has 50+ built-in responses across these categories:

| Category | Responses | Triggers |
|----------|-----------|----------|
| **Predefined** | 5 topics | Direct questions about EVA, CAM, PHIL, etc. |
| **Greeting** | 8 | hi, hello, hey, yo, howdy |
| **Help** | 6 | help, what can you do, who are you |
| **Farewell** | 8 | bye, goodbye, see you, cya |
| **Gratitude** | 8 | thanks, thank you, appreciate |
| **Acknowledgment** | 8 | ok, cool, great, nice, perfect |
| **Confusion** | 6 | what, huh, don't understand |
| **Unknown** | 10 | Anything outside scope |

## Project Structure

```
thoughtful-ai-agent/
├── agent.py          # Core agent logic with intent detection & semantic search
├── data.py           # Predefined Q&A + generic response library
├── main.py           # Rich CLI interface
├── requirements.txt  # Dependencies
├── .env.example      # Optional: OpenAI API key template
└── README.md         # Documentation
```

## Predefined Q&A Dataset

The agent knows about:

| Agent | Description |
|-------|-------------|
| **EVA** | Eligibility Verification - Real-time benefits verification |
| **CAM** | Claims Processing - Streamlined claims submission |
| **PHIL** | Payment Posting - Automated payment reconciliation |

Plus general questions about Thoughtful AI and its benefits.

## Tech Stack

- **Python 3.10+**
- **Rich** - Terminal UI
- **sentence-transformers** - Semantic embeddings
- **NumPy** - Vector operations
- **OpenAI** *(optional)* - Smart LLM fallback

## Testing

Run the automated test suite:

```bash
source venv/bin/activate
python3 -c "
from agent import ThoughtfulAIAgent
agent = ThoughtfulAIAgent()

tests = [
    'What does EVA do?',
    'What is CAM?',
    'How does PHIL work?',
    'hi',
    'what can you do',
    'thanks',
    'bye',
    'What is the weather?',
]

for q in tests:
    r = agent.respond(q)
    print(f'{r[\"source\"]}: {q}')
"
```

## License

MIT License - Built for Thoughtful AI coding challenge.

# 🤖 Thoughtful AI Customer Support Agent

A conversational AI Agent that answers questions about Thoughtful AI's healthcare automation agents using predefined responses, with generic fallback for unknown questions.

## Features

- ✅ **Semantic Search**: Matches user queries to predefined Q&A using sentence embeddings
- ✅ **Conversational CLI**: Beautiful terminal interface with Rich
- ✅ **Generic Fallback**: Polite responses for questions outside the dataset
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

### 3. Run the Agent

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

**Semantic Match:**
```
You: Tell me about CAM

Thoughtful AI Agent: CAM streamlines the submission and management of claims, 
improving accuracy, reducing manual intervention, and accelerating reimbursements.

✓ Predefined answer (confidence: 0.82)
```

**Generic Fallback:**
```
You: What's the weather like?

Thoughtful AI Agent: I'm not sure about that. I can help you with questions 
about Thoughtful AI's agents like EVA, CAM, and PHIL. Is there something 
specific about our healthcare automation agents I can help you with?

💬 Generic response
```

## How It Works

1. **Semantic Search**: User queries are embedded using `sentence-transformers` (all-MiniLM-L6-v2)
2. **Similarity Matching**: Cosine similarity compares query against predefined questions
3. **Threshold Check**: If similarity ≥ 0.55, return predefined answer
4. **Generic Fallback**: Otherwise, return a helpful generic response

## Project Structure

```
thoughtful-ai-agent/
├── agent.py          # Core agent logic with semantic search
├── data.py           # Predefined Q&A dataset with variations
├── main.py           # Rich CLI interface
├── requirements.txt  # Dependencies
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
    'Tell me about Thoughtful AI',
    'What are the benefits?',
    'What is the weather?',  # Generic fallback
]

for q in tests:
    r = agent.respond(q)
    print(f'{r[\"source\"]}: {q}')
"
```

## License

MIT License - Built for Thoughtful AI coding challenge.

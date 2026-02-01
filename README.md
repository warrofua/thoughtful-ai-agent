# 🤖 Thoughtful AI Customer Support Agent

A conversational AI Agent that answers questions about Thoughtful AI's healthcare automation agents using predefined responses with LLM fallback.

## Features

- ✅ **Semantic Search**: Matches user queries to predefined Q&A using sentence embeddings
- ✅ **Conversational CLI**: Beautiful terminal interface with Rich
- ✅ **LLM Fallback**: Uses OpenAI GPT for questions outside the dataset
- ✅ **Smart Matching**: Handles typos, variations, and rephrased questions
- ✅ **Error Handling**: Graceful handling of invalid inputs and API failures

## Quick Start

### 1. Clone and Navigate

```bash
cd thoughtful-ai-agent
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment (Optional)

For LLM fallback on unknown questions:

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

Without an API key, the agent will respond with a generic message for unknown questions.

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

✓ Predefined answer (confidence: 0.89)
```

**Semantic Match:**
```
You: Tell me about CAM

Thoughtful AI Agent: CAM streamlines the submission and management of claims, 
improving accuracy, reducing manual intervention, and accelerating reimbursements.

✓ Predefined answer (confidence: 0.82)
```

**LLM Fallback:**
```
You: What's the weather like?

Thoughtful AI Agent: I'm focused on helping you with Thoughtful AI's healthcare 
automation agents. For weather information, I'd recommend checking a weather app 
or website.

🤖 AI generated
```

## How It Works

1. **Semantic Search**: User queries are embedded using `sentence-transformers` (all-MiniLM-L6-v2)
2. **Similarity Matching**: Cosine similarity compares query against predefined questions
3. **Threshold Check**: If similarity ≥ 0.65, return predefined answer
4. **LLM Fallback**: Otherwise, use OpenAI GPT-3.5 for a generic response

## Project Structure

```
thoughtful-ai-agent/
├── agent.py          # Core agent logic with semantic search
├── data.py           # Predefined Q&A dataset
├── main.py           # Rich CLI interface
├── requirements.txt  # Dependencies
├── .env.example      # Environment template
└── README.md         # This file
```

## Predefined Q&A Dataset

The agent knows about:

| Agent | Description |
|-------|-------------|
| **EVA** | Eligibility Verification - Real-time benefits verification |
| **CAM** | Claims Processing - Streamlined claims submission |
| **PHIL** | Payment Posting - Automated payment reconciliation |

## Tech Stack

- **Python 3.10+**
- **Rich** - Terminal UI
- **sentence-transformers** - Semantic embeddings
- **OpenAI API** - LLM fallback
- **NumPy** - Vector operations

## Requirements

- Python 3.10 or higher
- (Optional) OpenAI API key for LLM fallback

## License

MIT License - Built for Thoughtful AI coding challenge.

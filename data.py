"""
Predefined Q&A dataset for Thoughtful AI Customer Support Agent.
Includes variations to handle different phrasings of the same question.
"""

PREDEFINED_QA = [
    # EVA - Eligibility Verification Agent
    {
        "question": "What does the eligibility verification agent (EVA) do?",
        "answer": "EVA automates the process of verifying a patient's eligibility and benefits information in real-time, eliminating manual data entry errors and reducing claim rejections.",
        "variations": [
            "What does EVA do?",
            "What is EVA?",
            "Tell me about EVA",
            "Explain EVA",
            "What is the eligibility verification agent?",
            "How does EVA work?",
        ]
    },
    # CAM - Claims Processing Agent
    {
        "question": "What does the claims processing agent (CAM) do?",
        "answer": "CAM streamlines the submission and management of claims, improving accuracy, reducing manual intervention, and accelerating reimbursements.",
        "variations": [
            "What is CAM?",
            "Tell me about CAM",
            "Explain CAM",
            "What is the claims processing agent?",
            "How does CAM work?",
            "What does CAM do?",
        ]
    },
    # PHIL - Payment Posting Agent
    {
        "question": "How does the payment posting agent (PHIL) work?",
        "answer": "PHIL automates the posting of payments to patient accounts, ensuring fast, accurate reconciliation of payments and reducing administrative burden.",
        "variations": [
            "What is PHIL?",
            "Tell me about PHIL",
            "Explain PHIL",
            "What is the payment posting agent?",
            "What does PHIL do?",
            "How does PHIL work?",
        ]
    },
    # General - About Thoughtful AI Agents
    {
        "question": "Tell me about Thoughtful AI's Agents.",
        "answer": "Thoughtful AI provides a suite of AI-powered automation agents designed to streamline healthcare processes. These include Eligibility Verification (EVA), Claims Processing (CAM), and Payment Posting (PHIL), among others.",
        "variations": [
            "What are Thoughtful AI agents?",
            "What agents do you have?",
            "Tell me about Thoughtful AI",
            "What is Thoughtful AI?",
            "What products do you offer?",
            "What services does Thoughtful AI provide?",
            "List your agents",
            "What AI agents are available?",
        ]
    },
    # Benefits
    {
        "question": "What are the benefits of using Thoughtful AI's agents?",
        "answer": "Using Thoughtful AI's Agents can significantly reduce administrative costs, improve operational efficiency, and reduce errors in critical processes like claims management and payment posting.",
        "variations": [
            "What are the benefits?",
            "Why should I use Thoughtful AI?",
            "What are the advantages?",
            "How can Thoughtful AI help me?",
            "What value do your agents provide?",
            "Tell me about the benefits",
            "Why choose Thoughtful AI?",
        ]
    }
]

# Build flattened list of all questions (main + variations) for semantic search
QUESTIONS = []
ANSWER_MAP = {}  # Maps normalized question to answer

for qa in PREDEFINED_QA:
    main_q = qa["question"]
    answer = qa["answer"]
    variations = qa.get("variations", [])
    
    # Add main question
    QUESTIONS.append(main_q)
    ANSWER_MAP[main_q] = answer
    
    # Add variations
    for var in variations:
        QUESTIONS.append(var)
        ANSWER_MAP[var] = answer

# For backward compatibility
ANSWERS = ANSWER_MAP

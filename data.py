"""
Predefined Q&A dataset for Thoughtful AI Customer Support Agent.
Includes variations to handle different phrasings of the same question.
Also includes facet-based keywords for functional queries.
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
        ],
        "facets": [  # Functional descriptions without naming the agent
            "verify eligibility",
            "check patient eligibility",
            "eligibility verification",
            "benefits verification",
            "check insurance eligibility",
            "verify benefits",
            "eligibility checks",
            "insurance verification",
            "patient eligibility",
            "real-time eligibility",
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
        ],
        "facets": [  # Functional descriptions without naming the agent
            "process claims",
            "claims submission",
            "submit claims",
            "manage claims",
            "claims management",
            "handle claims",
            "claims accuracy",
            "accelerate reimbursements",
            "claims processing",
            "claim rejections",
            "denied claims",
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
        ],
        "facets": [  # Functional descriptions without naming the agent
            "post payments",
            "payment posting",
            "patient payments",
            "payment reconciliation",
            "reconcile payments",
            "handle payments",
            "payment processing",
            "account reconciliation",
            "payment accuracy",
            "administrative burden",
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
        ],
        "facets": [
            "healthcare automation",
            "ai agents",
            "automation solutions",
            "healthcare ai",
            "streamline processes",
            "reduce manual work",
            "your solutions",
            "your products",
            "what do you offer",
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
        ],
        "facets": [
            "reduce costs",
            "save money",
            "operational efficiency",
            "reduce errors",
            "administrative costs",
            "improve accuracy",
            "faster processing",
            "save time",
            "return on investment",
            "roi",
        ]
    }
]

# Build flattened list of all questions (main + variations) for semantic search
QUESTIONS = []
ANSWER_MAP = {}  # Maps normalized question to answer

# Build facet-to-answer mapping
FACET_MAP = {}  # Maps facet keywords to answer

for qa in PREDEFINED_QA:
    main_q = qa["question"]
    answer = qa["answer"]
    variations = qa.get("variations", [])
    facets = qa.get("facets", [])
    
    # Add main question
    QUESTIONS.append(main_q)
    ANSWER_MAP[main_q] = answer
    
    # Add variations
    for var in variations:
        QUESTIONS.append(var)
        ANSWER_MAP[var] = answer
    
    # Add facets to facet map
    for facet in facets:
        FACET_MAP[facet.lower()] = answer

# For backward compatibility
ANSWERS = ANSWER_MAP


# ============================================================================
# GENERIC RESPONSE CATEGORIES
# ============================================================================

# Greeting responses - when user says hi, hello, hey, etc.
GREETING_RESPONSES = [
    "Hello! 👋 Welcome to Thoughtful AI Support. I'm here to help you with questions about our healthcare automation agents like EVA, CAM, and PHIL. What would you like to know?",
    "Hi there! Welcome to Thoughtful AI. I can tell you all about our AI agents that streamline healthcare processes. How can I assist you today?",
    "Hey! 👋 Thanks for reaching out to Thoughtful AI Support. I'm your friendly agent for questions about EVA, CAM, PHIL, and more. What can I help you with?",
    "Hello and welcome! I'm your Thoughtful AI support assistant. I specialize in our automation agents for healthcare. What would you like to learn about?",
    "Hi! Great to meet you. I'm here to answer questions about Thoughtful AI's agents - EVA for eligibility verification, CAM for claims processing, and PHIL for payment posting. What interests you?",
    "Welcome to Thoughtful AI! 🎉 I'm your support agent. Ask me anything about our healthcare automation solutions!",
    "Hello! I'm your virtual assistant for Thoughtful AI. I can explain how our agents help streamline healthcare operations. What would you like to explore?",
    "Hey there! Ready to learn about Thoughtful AI's automation agents? I'm here to help with any questions about EVA, CAM, PHIL, or our other solutions!",
]

# Help/Capability responses - when user asks what the agent can do
HELP_RESPONSES = [
    "I can help you with questions about Thoughtful AI's healthcare automation agents!\n\nI know about:\n• EVA - Eligibility Verification Agent\n• CAM - Claims Processing Agent\n• PHIL - Payment Posting Agent\n• General info about Thoughtful AI and our benefits\n\nWhat would you like to know about?",
    "Here's what I can help you with:\n\nOur Agents:\n• EVA - Automates patient eligibility verification\n• CAM - Streamlines claims processing\n• PHIL - Automates payment posting\n\nPlus general questions about Thoughtful AI and how our solutions benefit healthcare organizations.\n\nWhat interests you?",
    "I'm your Thoughtful AI support specialist! I can answer questions about:\n\n• What each agent (EVA, CAM, PHIL) does\n• How our automation works\n• The benefits of using Thoughtful AI\n• General information about our company\n\nWhat would you like to explore?",
    "Great question! I'm designed to help with:\n\n• Understanding EVA, CAM, and PHIL\n• Learning about Thoughtful AI's solutions\n• Discovering the benefits of healthcare automation\n• General support questions\n\nTry asking 'What is EVA?' or 'Tell me about CAM!'",
    "I can assist you with information about our AI-powered healthcare agents:\n\n• EVA - Real-time eligibility verification\n• CAM - Accurate claims processing\n• PHIL - Fast payment reconciliation\n\nPlus general questions about our company and solutions. What can I tell you about?",
    "Here's my expertise area:\n\nHealthcare Automation Agents:\n• EVA (Eligibility Verification)\n• CAM (Claims Processing)\n• PHIL (Payment Posting)\n\nPlus benefits, company info, and how we help healthcare organizations.\n\nWhat would you like to dive into?",
]

# Farewell/Goodbye responses - when user says bye, thanks, etc.
FAREWELL_RESPONSES = [
    "You're welcome! Thanks for chatting with Thoughtful AI Support. Have a great day! 👋",
    "Glad I could help! Feel free to come back if you have more questions about our agents. Take care!",
    "Thank you for reaching out! If you need more info about EVA, CAM, or PHIL later, I'm here. Goodbye! 👋",
    "You're welcome! Hope you learned something useful about Thoughtful AI. Have a wonderful day!",
    "Thanks for chatting! If you have more questions about our healthcare automation solutions, don't hesitate to ask. Bye!",
    "Appreciate you stopping by! Reach out anytime you need help with Thoughtful AI. Take care! 👋",
    "You're welcome! Wishing you success with your healthcare automation journey. Goodbye!",
    "Glad to assist! Come back anytime for questions about EVA, CAM, PHIL, or Thoughtful AI. Have a great one!",
]

# Gratitude responses - when user says thanks, thank you, etc.
GRATITUDE_RESPONSES = [
    "You're very welcome! 😊 Happy to help with any questions about our agents!",
    "My pleasure! Let me know if you need anything else about Thoughtful AI.",
    "Glad I could help! Feel free to ask if you want to dive deeper into any of our agents.",
    "You're welcome! Thanks for your interest in Thoughtful AI's solutions.",
    "Happy to assist! Is there anything else about EVA, CAM, or PHIL you'd like to know?",
    "Anytime! 😊 I'm here whenever you need info about our healthcare automation agents.",
    "You're most welcome! Excited to help you explore Thoughtful AI's capabilities.",
    "Glad to be of service! Reach out if you have more questions about our agents.",
]

# Unknown/Out-of-scope responses - when user asks something unrelated
UNKNOWN_RESPONSES = [
    "I'm not sure about that. I specialize in Thoughtful AI's healthcare automation agents like EVA, CAM, and PHIL. Is there something about those I can help you with?",
    "I don't have information on that topic. I'm specifically designed to answer questions about Thoughtful AI's agents (EVA, CAM, PHIL) and their benefits. How can I help you with those?",
    "That's outside my area of expertise. I focus on Thoughtful AI's healthcare automation solutions. Would you like to know about EVA, CAM, or PHIL instead?",
    "Hmm, I don't know about that. But I can tell you all about our AI agents for healthcare! Ask me about EVA, CAM, or PHIL?",
    "I wish I could help with that, but I'm specialized in Thoughtful AI's automation agents. Can I answer questions about EVA, CAM, or PHIL for you?",
    "That's not something I'm trained on. I'm your go-to for Thoughtful AI agent questions though! What would you like to know about EVA, CAM, or PHIL?",
    "I don't have the answer to that, but I'd love to help you learn about our healthcare automation agents! Which one interests you?",
    "Sorry, that's beyond my knowledge base. I stick to what I know best - Thoughtful AI's agents! Ask me about EVA, CAM, or PHIL?",
    "I'm focused on healthcare automation at Thoughtful AI. For that other topic, you might need a different resource. But hey, want to learn about our agents?",
    "That's a different topic than what I cover. I'm here for EVA, CAM, PHIL, and Thoughtful AI questions! What can I tell you?",
]

# Acknowledgment responses - when user says ok, got it, I see, etc.
ACKNOWLEDGMENT_RESPONSES = [
    "Great! Let me know if you have any other questions. I'm here to help!",
    "Got it! Feel free to ask if you want to learn more about any of our agents.",
    "Perfect! Is there anything else about Thoughtful AI you'd like to explore?",
    "Sounds good! I'm here if you need more info about EVA, CAM, or PHIL.",
    "Alright! Don't hesitate to reach out if you have more questions.",
    "Understood! Let me know if there's anything else I can clarify about our agents.",
    "Excellent! I'm ready to help with any follow-up questions you might have.",
    "Cool! Feel free to dig deeper into any topic about Thoughtful AI.",
]

# Confusion/Clarification responses - when user input is unclear
CONFUSION_RESPONSES = [
    "I'm not quite sure I understand. Could you rephrase that? I can help with questions about EVA, CAM, PHIL, or Thoughtful AI in general.",
    "Hmm, I didn't catch that. Try asking me about our agents like 'What is EVA?' or 'Tell me about CAM!'",
    "I want to help, but I'm not sure what you're asking. I know lots about Thoughtful AI's healthcare agents though!",
    "Could you try asking that a different way? I can answer questions about EVA, CAM, PHIL, or our company benefits.",
    "I'm a bit confused by that. But I can definitely help with: What is EVA? What does CAM do? How does PHIL work?",
    "Not sure I got that. I'm designed to help with Thoughtful AI agent questions. What would you like to know?",
]


# Keyword mappings for intent detection
INTENT_KEYWORDS = {
    "greeting": ["hi", "hello", "hey", "greetings", "howdy", "hiya", "yo", "sup", "morning", "afternoon", "evening"],
    "help": ["help", "what can you do", "what do you do", "capabilities", "what are you", "who are you", "features", "functions", "assist"],
    "farewell": ["bye", "goodbye", "see you", "cya", "later", "exit", "quit"],
    "gratitude": ["thanks", "thank you", "thx", "ty", "appreciate", "grateful", "cheers"],
    "acknowledgment": ["ok", "okay", "cool", "great", "good", "nice", "perfect", "sure", "alright"],
    "confusion": ["what", "huh", "confused", "don't understand", "dont understand"],
}

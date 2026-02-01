"""
CLI interface for Thoughtful AI Customer Support Agent.

This module provides the terminal-based user interface using the Rich library.
It includes:
- Startup animations for professional first impression
- Chat-style message bubbles for conversation
- Status indicators and source labels for transparency
- Command handling (/help, /examples, /quit)
- Conversation summary on exit

Usage:
    python main.py

Commands:
    /help      - Show welcome message
    /examples  - Show example questions
    /quit      - Exit with conversation summary
"""

import sys
import time
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from rich import box
from rich.align import Align
from rich.rule import Rule

# Import the core agent
from agent import ThoughtfulAIAgent


def create_startup_animation(console: Console) -> None:
    """
    Create a subtle startup animation sequence.
    
    This function creates a professional first impression by:
    1. Clearing the screen for a clean start
    2. Showing a centered logo with branding
    3. Displaying loading steps with delays for visual interest
    4. Using Rich's status indicators for polish
    
    The animation is designed to be subtle and quick (< 2 seconds total)
    to not delay the user unnecessarily.
    
    Args:
        console: Rich Console instance for output
    """
    # Clear screen for clean start
    console.clear()
    
    # Animation 1: Brand logo
    # Using a simple box with hearts for visual appeal
    logo_text = Text()
    logo_text.append("╭─────────────────────────────╮\n", style="dim blue")
    logo_text.append("│                             │\n", style="dim blue")
    logo_text.append("│   💙  Thoughtful AI  💙    │\n", style="bold cyan")
    logo_text.append("│                             │\n", style="dim blue")
    logo_text.append("│   Customer Support Agent    │\n", style="dim")
    logo_text.append("│                             │\n", style="dim blue")
    logo_text.append("╰─────────────────────────────╯", style="dim blue")
    
    console.print(Align.center(logo_text))
    console.print()
    
    # Animation 2: Loading indicator
    # Using Rich's built-in status spinner
    with console.status("[cyan]Initializing...[/cyan]", spinner="dots"):
        time.sleep(0.8)
    
    # Animation 3: Connection sequence
    # Sequential messages create a sense of progress
    steps = [
        ("[dim]→ Loading configuration...[/dim]", 0.2),
        ("[dim]→ Connecting to knowledge base...[/dim]", 0.3),
        ("[dim]→ Initializing semantic search...[/dim]", 0.4),
    ]
    
    for message, delay in steps:
        console.print(Align.center(message))
        time.sleep(delay)
    
    console.print()


def create_welcome_panel() -> Panel:
    """
    Create the welcome panel with instructions.
    
    This panel is shown at startup and when the user types /help.
    It provides:
    - Welcome message with emoji
    - List of topics the agent can help with
    - Available commands
    
    Returns:
        Rich Panel with styled welcome content
    """
    welcome_text = Text()
    welcome_text.append("👋 ", style="bold yellow")
    welcome_text.append("Welcome to Thoughtful AI Support!\n\n", style="bold cyan")
    welcome_text.append("I can help you with questions about:\n", style="white")
    welcome_text.append("  • EVA (Eligibility Verification Agent)\n", style="green")
    welcome_text.append("  • CAM (Claims Processing Agent)\n", style="green")
    welcome_text.append("  • PHIL (Payment Posting Agent)\n", style="green")
    welcome_text.append("  • General questions about Thoughtful AI\n\n", style="green")
    welcome_text.append("Commands:\n", style="bold")
    welcome_text.append("  • Type your question and press Enter\n", style="dim")
    welcome_text.append("  • Type ", style="dim")
    welcome_text.append("/quit", style="bold red")
    welcome_text.append(" or ", style="dim")
    welcome_text.append("/exit", style="bold red")
    welcome_text.append(" to exit\n", style="dim")
    welcome_text.append("  • Type ", style="dim")
    welcome_text.append("/help", style="bold yellow")
    welcome_text.append(" to see this message again\n", style="dim")
    welcome_text.append("  • Type ", style="dim")
    welcome_text.append("/examples", style="bold magenta")
    welcome_text.append(" to see example questions", style="dim")
    
    return Panel(
        Align.center(welcome_text),
        title="[bold blue]Thoughtful AI Agent[/bold blue]",
        border_style="blue",
        box=box.ROUNDED,
        padding=(1, 2)
    )


def create_examples_panel() -> Panel:
    """
    Create a panel showing example questions.
    
    This panel is shown when the user types /examples.
    It provides concrete examples of questions the agent can answer,
    color-coded by topic for visual distinction.
    
    Returns:
        Rich Panel with example questions
    """
    # Define examples with emojis and colors for visual appeal
    examples = [
        ("💙", "What does EVA do?", "cyan"),          # EVA - blue
        ("💚", "What is CAM?", "green"),              # CAM - green
        ("💛", "How does PHIL work?", "yellow"),      # PHIL - yellow
        ("🤍", "Tell me about Thoughtful AI", "white"),  # General - white
        ("💜", "What are the benefits?", "magenta"),  # Benefits - purple
        ("👋", "Hi", "bright_black"),                 # Greeting
        ("❓", "What can you do?", "bright_black"),   # Help
    ]
    
    text = Text()
    text.append("Try asking me:\n\n", style="bold")
    
    for emoji, question, color in examples:
        text.append(f"{emoji} ", style=color)
        text.append(f"{question}\n", style=color)
    
    return Panel(
        text,
        title="[bold magenta]Example Questions[/bold magenta]",
        border_style="magenta",
        box=box.ROUNDED,
        padding=(1, 2)
    )


def create_status_bar(agent) -> Text:
    """
    Create a status bar showing agent configuration.
    
    Displays:
    - Online status (always shown)
    - Enhanced mode indicator (if OpenAI is enabled)
    
    Args:
        agent: The ThoughtfulAIAgent instance
        
    Returns:
        Rich Text object with status indicators
    """
    status = Text()
    status.append("🟢 ", style="green")
    status.append("Online", style="dim green")
    
    # Show enhanced mode if OpenAI is available
    if agent.openai_enabled:
        status.append("  •  ", style="dim")
        status.append("🤖 ", style="magenta")
        status.append("Enhanced", style="dim magenta")
    
    return status


def format_user_message(message: str) -> Panel:
    """
    Format user message as a chat bubble.
    
    Creates a right-aligned panel with the user's message,
    styled with a green border to distinguish from agent responses.
    
    Args:
        message: The user's input message
        
    Returns:
        Rich Panel with styled user message
    """
    return Panel(
        Text(message, style="white"),
        title="[bold green]You[/bold green]",
        border_style="green",
        box=box.ROUNDED,
        width=80
    )


def format_agent_message(response: dict) -> Panel:
    """
    Format agent response as a chat bubble with source indicator.
    
    Creates a left-aligned panel with:
    - The agent's response text
    - A subtitle indicating the source (predefined, generic, LLM, etc.)
    - Color-coded confidence scores for predefined answers
    
    Args:
        response: Dictionary containing 'response', 'source', and optionally 'confidence'
        
    Returns:
        Rich Panel with styled agent message and source indicator
    """
    content_text = response["response"]
    
    # Use plain text for reliable formatting (Markdown can have wrapping issues)
    content = Text(content_text, style="white")
    
    # Build source indicator subtitle
    footer_text = Text()
    source = response["source"]
    
    if source == "predefined":
        # Predefined answer from dataset
        footer_text.append("✓ ", style="bold green")
        footer_text.append(f"Predefined answer", style="dim green")
        
        # Add color-coded confidence score
        if response.get("confidence"):
            confidence = response["confidence"]
            # Color code: green (high), yellow (medium), red (low)
            if confidence >= 0.9:
                conf_style = "green"
            elif confidence >= 0.7:
                conf_style = "yellow"
            else:
                conf_style = "red"
            footer_text.append(f" (confidence: ", style="dim")
            footer_text.append(f"{confidence:.2f}", style=conf_style)
            footer_text.append(")", style="dim")
    
    elif source == "llm":
        # OpenAI-generated response
        footer_text.append("🤖 ", style="bold magenta")
        footer_text.append("AI enhanced", style="dim magenta")
    
    elif source.startswith("generic-"):
        # Generic intent-based response
        intent = source.replace("generic-", "")
        emoji_map = {
            "greeting": "👋",
            "help": "❓",
            "farewell": "👋",
            "gratitude": "🙏",
            "ack": "✓",
            "confusion": "🤔",
            "unknown": "💬",
        }
        emoji = emoji_map.get(intent, "💬")
        footer_text.append(f"{emoji} ", style="bold blue")
        footer_text.append(f"{intent.capitalize()} response", style="dim blue")
    
    else:
        # Error or other unexpected source
        footer_text.append("⚠ ", style="bold yellow")
        footer_text.append("System message", style="dim yellow")
    
    return Panel(
        content,
        title="[bold cyan]Thoughtful AI Agent[/bold cyan]",
        subtitle=footer_text,
        border_style="cyan",
        box=box.ROUNDED,
        width=80
    )


def show_typing_indicator(console: Console) -> None:
    """
    Show a brief typing indicator animation.
    
    This creates the illusion that the agent is "thinking" before responding,
    making the interaction feel more natural and conversational.
    
    Args:
        console: Rich Console instance for output
    """
    with console.status("[bold cyan]🤔 Thinking...[/bold cyan]", spinner="dots2"):
        time.sleep(0.4)  # Brief pause for effect


def show_exit_animation(console: Console) -> None:
    """
    Show a subtle exit animation.
    
    Provides visual feedback that the session is closing gracefully.
    
    Args:
        console: Rich Console instance for output
    """
    console.print()
    with console.status("[dim]Closing session...[/dim]", spinner="moon"):
        time.sleep(0.5)


def main():
    """
    Main entry point for the CLI.
    
    This function orchestrates the entire user interaction:
    1. Shows startup animation
    2. Displays welcome message
    3. Initializes the agent
    4. Enters main conversation loop
    5. Handles commands and user input
    6. Shows conversation summary on exit
    
    The loop continues until the user types /quit or presses Ctrl+C.
    """
    console = Console()
    
    # Show startup animation for professional first impression
    create_startup_animation(console)
    
    # Display welcome panel with instructions
    console.print(create_welcome_panel())
    console.print()
    
    # Initialize the agent
    # This loads the embedding model and pre-computes question embeddings
    try:
        agent = ThoughtfulAIAgent()
    except Exception as e:
        console.print(Panel(
            f"[bold red]Error initializing agent:[/bold red] {str(e)}\n"
            "Please ensure you have installed all dependencies:\n"
            "  pip install -r requirements.txt",
            title="Error",
            border_style="red"
        ))
        sys.exit(1)
    
    # Show status bar (Online + Enhanced if OpenAI enabled)
    console.print(Align.center(create_status_bar(agent)))
    console.print()
    
    # Track conversation history for summary on exit
    conversation_history = []
    
    # =========================================================================
    # MAIN CONVERSATION LOOP
    # =========================================================================
    while True:
        try:
            # Get user input with styled prompt
            user_input = Prompt.ask("[bold green]You[/bold green]")
            
            # -----------------------------------------------------------------
            # COMMAND HANDLING
            # -----------------------------------------------------------------
            
            # Exit commands
            if user_input.lower() in ['/quit', '/exit', 'quit', 'exit']:
                # Show conversation summary if there was a conversation
                if conversation_history:
                    console.print()
                    console.print(Rule("[dim]Conversation Summary[/dim]", style="dim"))
                    for item in conversation_history[-5:]:  # Last 5 exchanges
                        console.print(f"[dim]• {item[:60]}...[/dim]")
                
                # Show exit animation and goodbye message
                console.print()
                show_exit_animation(console)
                console.print(Panel(
                    "[italic]Thank you for using Thoughtful AI Support. Goodbye! 👋[/italic]",
                    border_style="dim"
                ))
                break
            
            # Help command
            if user_input.lower() in ['/help', 'help']:
                console.print(create_welcome_panel())
                console.print()
                continue
            
            # Examples command
            if user_input.lower() in ['/examples', '/example', 'examples']:
                console.print(create_examples_panel())
                console.print()
                continue
            
            # Skip empty input
            if not user_input.strip():
                continue
            
            # -----------------------------------------------------------------
            # PROCESS USER MESSAGE
            # -----------------------------------------------------------------
            
            # Display user message in a chat bubble
            console.print(format_user_message(user_input))
            
            # Show typing indicator for natural feel
            show_typing_indicator(console)
            
            # Get agent response
            response = agent.respond(user_input)
            
            # Track in conversation history
            conversation_history.append(f"Q: {user_input}")
            conversation_history.append(f"A: {response['response'][:50]}...")
            
            # Display agent response
            console.print(format_agent_message(response))
            console.print()
            
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            console.print("\n")
            show_exit_animation(console)
            console.print(Panel(
                "[italic]Interrupted. Goodbye! 👋[/italic]",
                border_style="dim"
            ))
            break
        except Exception as e:
            # Handle unexpected errors
            console.print(Panel(
                f"[bold red]Error:[/bold red] {str(e)}",
                border_style="red"
            ))


if __name__ == "__main__":
    # Entry point when script is run directly
    main()

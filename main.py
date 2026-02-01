"""
CLI interface for Thoughtful AI Customer Support Agent.
Uses Rich for beautiful terminal output.
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
from rich.live import Live
from rich.layout import Layout
from rich.spinner import Spinner

from agent import ThoughtfulAIAgent


def typing_effect(console: Console, text: str, style: str = "", delay: float = 0.01):
    """Display text with a typing animation effect."""
    displayed = ""
    for char in text:
        displayed += char
        console.print(f"\r{displayed}", style=style, end="")
        time.sleep(delay)
    console.print()  # New line at end


def create_startup_animation(console: Console) -> None:
    """Create a subtle startup animation sequence."""
    # Clear screen for clean start
    console.clear()
    
    # Animation 1: Brand logo fade-in with typing
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
    
    # Animation 2: Loading bar with pulses
    with console.status("[cyan]Initializing...[/cyan]", spinner="dots") as status:
        time.sleep(0.8)
    
    # Animation 3: Connection sequence (quick, subtle)
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
    """Create the welcome panel with instructions."""
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
    """Create a panel showing example questions."""
    examples = [
        ("💙", "What does EVA do?", "cyan"),
        ("💚", "What is CAM?", "green"),
        ("💛", "How does PHIL work?", "yellow"),
        ("🤍", "Tell me about Thoughtful AI", "white"),
        ("💜", "What are the benefits?", "magenta"),
        ("👋", "Hi", "bright_black"),
        ("❓", "What can you do?", "bright_black"),
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
    """Create a status bar showing agent configuration."""
    status = Text()
    status.append("🟢 ", style="green")
    status.append("Online", style="dim green")
    
    if agent.openai_enabled:
        status.append("  •  ", style="dim")
        status.append("🤖 ", style="magenta")
        status.append("Enhanced", style="dim magenta")
    
    return status


def format_user_message(message: str) -> Panel:
    """Format user message as a chat bubble with slide-in effect."""
    return Panel(
        Text(message, style="white"),
        title="[bold green]You[/bold green]",
        border_style="green",
        box=box.ROUNDED,
        width=80
    )


def format_agent_message(response: dict) -> Panel:
    """Format agent response as a chat bubble with source indicator."""
    content_text = response["response"]
    
    # Use plain text with preserved newlines for reliable formatting
    content = Text(content_text, style="white")
    
    # Add source indicator
    footer_text = Text()
    source = response["source"]
    
    if source == "predefined":
        footer_text.append("✓ ", style="bold green")
        footer_text.append(f"Predefined answer", style="dim green")
        if response["confidence"]:
            # Color-code confidence
            confidence = response["confidence"]
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
        footer_text.append("🤖 ", style="bold magenta")
        footer_text.append("AI enhanced", style="dim magenta")
    elif source.startswith("generic-"):
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


def show_typing_indicator(console: Console):
    """Show a brief typing indicator with animation."""
    with console.status("[bold cyan]🤔 Thinking...[/bold cyan]", spinner="dots2"):
        time.sleep(0.4)  # Brief pause for effect


def show_exit_animation(console: Console):
    """Show a subtle exit animation."""
    console.print()
    with console.status("[dim]Closing session...[/dim]", spinner="moon"):
        time.sleep(0.5)


def main():
    """Main entry point for the CLI."""
    console = Console()
    
    # Show startup animation
    create_startup_animation(console)
    
    # Print welcome message
    console.print(create_welcome_panel())
    console.print()
    
    # Initialize agent (with its own loading message)
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
    
    # Show status bar
    console.print(Align.center(create_status_bar(agent)))
    console.print()
    
    # Conversation history for summary
    conversation_history = []
    
    # Main conversation loop
    while True:
        try:
            # Get user input
            user_input = Prompt.ask("[bold green]You[/bold green]")
            
            # Handle commands
            if user_input.lower() in ['/quit', '/exit', 'quit', 'exit']:
                # Show conversation summary before exit
                if conversation_history:
                    console.print()
                    console.print(Rule("[dim]Conversation Summary[/dim]", style="dim"))
                    for item in conversation_history[-5:]:  # Last 5 exchanges
                        console.print(f"[dim]• {item[:60]}...[/dim]")
                
                console.print()
                show_exit_animation(console)
                console.print(Panel(
                    "[italic]Thank you for using Thoughtful AI Support. Goodbye! 👋[/italic]",
                    border_style="dim"
                ))
                break
            
            if user_input.lower() in ['/help', 'help']:
                console.print(create_welcome_panel())
                console.print()
                continue
            
            if user_input.lower() in ['/examples', '/example', 'examples']:
                console.print(create_examples_panel())
                console.print()
                continue
            
            if not user_input.strip():
                continue
            
            # Display user message
            console.print(format_user_message(user_input))
            
            # Show typing indicator
            show_typing_indicator(console)
            
            # Get and display agent response
            response = agent.respond(user_input)
            
            # Add to history
            conversation_history.append(f"Q: {user_input}")
            conversation_history.append(f"A: {response['response'][:50]}...")
            
            console.print(format_agent_message(response))
            console.print()
            
        except KeyboardInterrupt:
            console.print("\n")
            show_exit_animation(console)
            console.print(Panel(
                "[italic]Interrupted. Goodbye! 👋[/italic]",
                border_style="dim"
            ))
            break
        except Exception as e:
            console.print(Panel(
                f"[bold red]Error:[/bold red] {str(e)}",
                border_style="red"
            ))


if __name__ == "__main__":
    main()

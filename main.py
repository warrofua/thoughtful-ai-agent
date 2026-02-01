"""
CLI interface for Thoughtful AI Customer Support Agent.
Uses Rich for beautiful terminal output.
"""

import sys
from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.prompt import Prompt
from rich import box

from agent import ThoughtfulAIAgent


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
    welcome_text.append(" to see this message again", style="dim")
    
    return Panel(
        welcome_text,
        title="[bold blue]Thoughtful AI Agent[/bold blue]",
        border_style="blue",
        box=box.ROUNDED,
        padding=(1, 2)
    )


def format_user_message(message: str) -> Panel:
    """Format user message as a chat bubble."""
    return Panel(
        Text(message, style="white"),
        title="[bold green]You[/bold green]",
        border_style="green",
        box=box.ROUNDED,
        width=80
    )


def format_agent_message(response: dict) -> Panel:
    """Format agent response as a chat bubble with source indicator."""
    content = Text(response["response"], style="white")
    
    # Add source indicator
    footer_text = Text()
    source = response["source"]
    
    if source == "predefined":
        footer_text.append("✓ ", style="bold green")
        footer_text.append(f"Predefined answer", style="dim green")
        if response["confidence"]:
            footer_text.append(f" (confidence: {response['confidence']:.2f})", style="dim")
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


def main():
    """Main entry point for the CLI."""
    console = Console()
    
    # Print welcome message
    console.print(create_welcome_panel())
    console.print()
    
    # Initialize agent
    try:
        agent = ThoughtfulAIAgent()
        # Show OpenAI status (subtle indicator)
        if agent.openai_enabled:
            console.print("[dim italic]🤖 Enhanced responses enabled[/dim italic]")
            console.print()
    except Exception as e:
        console.print(Panel(
            f"[bold red]Error initializing agent:[/bold red] {str(e)}\n"
            "Please ensure you have installed all dependencies:\n"
            "  pip install -r requirements.txt",
            title="Error",
            border_style="red"
        ))
        sys.exit(1)
    
    console.print()
    
    # Main conversation loop
    while True:
        try:
            # Get user input
            user_input = Prompt.ask("[bold green]You[/bold green]")
            
            # Handle commands
            if user_input.lower() in ['/quit', '/exit', 'quit', 'exit']:
                console.print(Panel(
                    "[italic]Thank you for using Thoughtful AI Support. Goodbye! 👋[/italic]",
                    border_style="dim"
                ))
                break
            
            if user_input.lower() in ['/help', 'help']:
                console.print(create_welcome_panel())
                console.print()
                continue
            
            if not user_input.strip():
                continue
            
            # Display user message
            console.print(format_user_message(user_input))
            
            # Get and display agent response
            response = agent.respond(user_input)
            console.print(format_agent_message(response))
            console.print()
            
        except KeyboardInterrupt:
            console.print("\n")
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

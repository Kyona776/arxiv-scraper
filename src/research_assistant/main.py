"""
This module serves as the command-line interface (CLI) entry point
for the Research Assistant functionality, powered by Typer and PocketFlow.
"""

import typer
import logging
from rich.console import Console
from rich.logging import RichHandler
from .flows import create_interactive_search_flow

# ==============================================================================
# Setup Logging and CLI App
# ==============================================================================

# Configure logging to be rich and informative
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

# Initialize Typer app and Rich console
app = typer.Typer(name="research", help="AI-powered Research Assistant CLI")
console = Console()


# ==============================================================================
# CLI Commands
# ==============================================================================


@app.command()
def start():
    """
    Starts a new research project by launching the fully interactive
    search and refinement workflow.
    """
    console.print(
        f"[bold green]🚀 Starting new interactive research session...[/bold green]"
    )
    console.print("You will be guided to build a powerful search query.")

    # 1. Create the new interactive research flow
    interactive_flow = create_interactive_search_flow()

    # 2. Prepare an empty shared store. The flow will populate it.
    shared_store = {
        "query": {},  # This will be populated by the interactive nodes
    }

    try:
        # 3. Run the flow
        interactive_flow.run(shared_store)

        # 4. Print the results
        project_path = shared_store.get("project_path")
        console.print(
            "\n[bold green]✅ Research project successfully created![/bold green]"
        )
        console.print(
            f"   Project Path: [link=file://{project_path}]{project_path}[/link]"
        )

    except Exception as e:
        console.print(
            f"\n[bold red]❌ An error occurred during the research flow.[/bold red]"
        )
        logging.exception("Flow execution failed.")



@app.command()
def open(
    project_name: str = typer.Argument(..., help="The name of the project to open."),
):
    """
    Opens an existing project to start an interactive refinement session.
    (This is a placeholder for future implementation)
    """
    console.print(
        f"[bold yellow]🚧 Opening project '{project_name}'... (Not yet implemented)[/bold yellow]"
    )
    # Here you would initialize and run the `create_interactive_refinement_flow`


if __name__ == "__main__":
    app()

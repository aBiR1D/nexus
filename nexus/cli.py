"""Nexus CLI — run, test, and inspect agents from the terminal."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from nexus import __version__

app = typer.Typer(
    name="nexus",
    help="⚡ The Agent Runtime — build reliable, composable AI agents",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()

BANNER = r"""
[bold cyan]
  _   _
 | \ | | _____  ___   _ ___
 |  \| |/ _ \ \/ / | | / __|
 | |\  |  __/>  <| |_| \__ \
 |_| \_|\___/_/\_\\__,_|___/
[/bold cyan]
[dim]The Agent Runtime • v{version}[/dim]
""".format(version=__version__)


@app.command()
def run(
    agent_path: str = typer.Argument(..., help="Agent module path (e.g. my_agents:reviewer)"),
    input_json: str = typer.Option("{}", "--input", "-i", help="JSON input for the agent"),
    provider: str = typer.Option("anthropic", "--provider", "-p", help="LLM provider"),
    model: str = typer.Option("", "--model", "-m", help="LLM model"),
    trace: bool = typer.Option(True, "--trace/--no-trace", help="Show execution trace"),
):
    """⚡ Run an agent."""
    console.print(BANNER)

    from nexus.runtime import Runtime
    from nexus.llm.provider import LLMProvider

    module_path, _, agent_name = agent_path.rpartition(":")
    if not module_path:
        module_path = agent_name
        agent_name = ""

    try:
        import importlib
        mod = importlib.import_module(module_path)
        if agent_name:
            agent_obj = getattr(mod, agent_name)
        else:
            from nexus.agent import Agent
            agents = [v for v in vars(mod).values() if isinstance(v, Agent)]
            if not agents:
                console.print("[red]No agents found in module[/red]")
                raise typer.Exit(1)
            agent_obj = agents[0]
    except Exception as e:
        console.print(f"[red]Failed to load agent: {e}[/red]")
        raise typer.Exit(1)

    kwargs = json.loads(input_json)
    llm_kwargs = {"provider": provider}
    if model:
        llm_kwargs["model"] = model

    runtime = Runtime(llm=LLMProvider(**llm_kwargs))

    console.print(f"[cyan]Running agent:[/cyan] {agent_obj.name}")
    result = asyncio.run(runtime.run(agent_obj, **kwargs))

    if result.succeeded:
        console.print(Panel(str(result.output), title="✅ Output", border_style="green"))
    else:
        console.print(Panel(str(result.error), title="❌ Failed", border_style="red"))

    if trace:
        console.print()
        tree_str = runtime.trace.print_tree()
        console.print(Panel(tree_str, title="📊 Trace", border_style="blue"))

    stats = Table(show_header=False, box=None, padding=(0, 2))
    stats.add_row("Duration", f"{result.duration_ms:.0f}ms")
    stats.add_row("Tokens", str(result.tokens_used))
    stats.add_row("Guardrails", f"{result.guardrails_passed}/{result.guardrails_checked}")
    console.print(Panel(stats, title="📈 Stats", border_style="dim"))


@app.command()
def agents():
    """📋 List built-in agents."""
    console.print(BANNER)

    from nexus.agents import CoderAgent, ReviewerAgent, ResearcherAgent

    table = Table(title="Built-in Agents", show_lines=True)
    table.add_column("Agent", style="cyan")
    table.add_column("Description")
    table.add_column("Tags", style="dim")

    for AgentClass in [CoderAgent, ReviewerAgent, ResearcherAgent]:
        a = AgentClass()
        table.add_row(a.name, a.spec.description, ", ".join(a.spec.tags))

    console.print(table)
    console.print("\n[dim]Create your own: @nexus.agent(name='my_agent')[/dim]")


@app.command()
def mcp():
    """🔌 Start the MCP server for Claude CLI."""
    from nexus.mcp.server import mcp as mcp_server
    console.print("[dim]Starting Nexus MCP server...[/dim]", err=True)
    mcp_server.run()


@app.command()
def version():
    """Show version."""
    console.print(f"Nexus v{__version__}")


@app.command()
def init(
    path: Path = typer.Argument(Path("."), help="Project directory"),
):
    """🏗️  Initialize a new Nexus agent project."""
    console.print(BANNER)

    agent_file = path / "agents.py"
    if not agent_file.exists():
        agent_file.write_text('''"""My Nexus agents."""

import nexus
from nexus.context import Context


@nexus.agent(name="hello", description="A simple greeting agent")
async def hello(name: str, ctx: Context) -> str:
    """Greet someone using AI."""
    return await ctx.llm.generate(
        f"Write a warm, creative greeting for {name}. Keep it under 2 sentences."
    )


@nexus.agent(name="summarizer", description="Summarizes text")
async def summarize(text: str, ctx: Context) -> str:
    """Summarize text concisely."""
    past = await ctx.memory.recall("summarizer", "summary_style")
    style_hint = ""
    if past:
        style_hint = f"\\nPreferred style: {past[0]}"

    result = await ctx.llm.generate(
        f"Summarize this text in 2-3 sentences:{style_hint}\\n\\n{text}"
    )

    await ctx.memory.remember("summarizer", "summary_style", "concise and clear")
    return result
''')
        console.print(f"[green]✅ Created {agent_file}[/green]")

    test_file = path / "test_agents.py"
    if not test_file.exists():
        test_file.write_text('''"""Tests for my agents."""

import pytest
from nexus.testing import AgentTest
from agents import hello, summarize


@pytest.mark.asyncio
async def test_hello():
    test = AgentTest(mock_responses=["Hello, World! Welcome aboard!"])
    result = await test.run(hello, name="World")
    assert result.succeeded
    test.mock.assert_called(times=1)
    test.mock.assert_prompt_contains("World")


@pytest.mark.asyncio
async def test_summarize():
    test = AgentTest(mock_responses=["This is a summary."])
    result = await test.run(summarize, text="A very long text...")
    assert result.succeeded
    assert result.output == "This is a summary."
''')
        console.print(f"[green]✅ Created {test_file}[/green]")

    env_file = path / ".env"
    if not env_file.exists():
        env_file.write_text("ANTHROPIC_API_KEY=\nNEXUS_AI_PROVIDER=anthropic\n")
        console.print(f"[green]✅ Created {env_file}[/green]")

    console.print(
        Panel(
            "Your Nexus project is ready!\\n\\n"
            "[cyan]nexus run agents:hello -i '{\"name\": \"World\"}'[/cyan]  — Run an agent\\n"
            "[cyan]pytest test_agents.py[/cyan]                          — Test agents\\n"
            "[cyan]nexus agents[/cyan]                                   — List built-in agents",
            title="🚀 Next Steps",
            border_style="green",
        )
    )


if __name__ == "__main__":
    app()

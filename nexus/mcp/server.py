"""Nexus MCP Server — exposes agent tools to Claude CLI."""

from __future__ import annotations

import asyncio
import json
from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "nexus",
    instructions=(
        "Nexus is an AI Agent Runtime. Use these tools to create, run, compose, "
        "and test AI agents. Agents have persistent memory, guardrails, and full "
        "execution tracing."
    ),
)


@mcp.tool()
def run_builtin_agent(
    agent_type: Annotated[str, "Built-in agent: 'coder', 'reviewer', or 'researcher'"],
    input_json: Annotated[str, "JSON input for the agent, e.g. {\"task\": \"write a sort function\", \"language\": \"python\"}"],
) -> str:
    """Run a built-in Nexus agent (coder, reviewer, or researcher).

    Examples:
        - Coder: {"task": "Write a Python function to merge two sorted lists", "language": "python"}
        - Reviewer: {"code": "def foo(x): return x+1", "language": "python"}
        - Researcher: {"question": "What are the best practices for API design?"}
    """
    from nexus.agents import CoderAgent, ReviewerAgent, ResearcherAgent
    from nexus.runtime import Runtime

    agents_map = {
        "coder": CoderAgent,
        "reviewer": ReviewerAgent,
        "researcher": ResearcherAgent,
    }

    if agent_type not in agents_map:
        return json.dumps({"error": f"Unknown agent type: {agent_type}. Available: {list(agents_map.keys())}"})

    agent = agents_map[agent_type]()
    kwargs = json.loads(input_json)
    runtime = Runtime()

    try:
        result = asyncio.run(runtime.run(agent, **kwargs))
        output = result.output
        if hasattr(output, "model_dump"):
            output = output.model_dump()

        return json.dumps({
            "status": result.status.value,
            "output": output,
            "duration_ms": round(result.duration_ms, 2),
            "tokens_used": result.tokens_used,
            "trace": runtime.trace.print_tree(),
        }, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def list_agents() -> str:
    """List all available built-in Nexus agents and their capabilities."""
    from nexus.agents import CoderAgent, ReviewerAgent, ResearcherAgent

    agents = []
    for AgentClass in [CoderAgent, ReviewerAgent, ResearcherAgent]:
        a = AgentClass()
        agents.append({
            "name": a.name,
            "description": a.spec.description,
            "tags": a.spec.tags,
        })

    return json.dumps({"agents": agents}, indent=2)


@mcp.tool()
def create_agent_code(
    name: Annotated[str, "Name for the new agent"],
    description: Annotated[str, "What the agent should do"],
    input_fields: Annotated[str, "Comma-separated input field names, e.g. 'task,language'"],
    output_type: Annotated[str, "Output type: 'str', 'structured', or 'dict'"] = "str",
) -> str:
    """Generate Python code for a new Nexus agent.

    Returns ready-to-use code for a custom agent with the @nexus.agent decorator,
    including proper context handling, memory integration, and type hints.
    """
    fields = [f.strip() for f in input_fields.split(",")]
    params = ", ".join(f"{f}: str" for f in fields)

    code = f'''import nexus
from nexus.context import Context


@nexus.agent(name="{name}", description="{description}")
async def {name}({params}, ctx: Context) -> str:
    """{description}"""
    # Check memory for past patterns
    past = await ctx.memory.recall("{name}", "{fields[0]}")
    context_hint = ""
    if past:
        items = [str(p.value) if hasattr(p, "value") else str(p) for p in past[:3]]
        context_hint = "\\nContext from past runs:\\n" + "\\n".join(items)

    result = await ctx.llm.generate(
        f"{{context_hint}}\\n{description}\\n"
        + "\\n".join(f"{f}: {{{f}}}" for f in fields)
    )

    # Store learnings
    await ctx.memory.remember("{name}", "{fields[0]}", result[:200])

    return result
'''

    test_code = f'''import pytest
from nexus.testing import AgentTest
from agents import {name}


@pytest.mark.asyncio
async def test_{name}():
    test = AgentTest(mock_responses=["Mock output"])
    result = await test.run({name}, {", ".join(f'{f}="test"' for f in fields)})
    assert result.succeeded
    test.mock.assert_called(times=1)
'''

    return json.dumps({
        "agent_code": code,
        "test_code": test_code,
        "usage": f'nexus run agents:{name} -i \'{json.dumps({f: "example" for f in fields})}\'',
    }, indent=2)


@mcp.tool()
def explain_nexus() -> str:
    """Explain what Nexus is and how to use it."""
    return json.dumps({
        "what": "Nexus is an Agent Runtime — like Docker for AI agents",
        "key_concepts": {
            "@nexus.agent": "Decorator to turn any async function into a reliable agent",
            "Context": "Injected into every agent — provides LLM, memory, trace, guardrails",
            "Memory": "Persistent learning across invocations (SQLite-backed)",
            "Guardrails": "Safety constraints (no PII, rate limits, content validation)",
            "Compose": "Wire agents together: Pipeline (sequential), Parallel, Router, Graph (DAG)",
            "Trace": "Full execution tree with timing, errors, and events",
            "Testing": "Mock LLMs for deterministic agent tests",
        },
        "quick_start": [
            "pip install nexus-agents",
            "nexus init",
            "# Edit agents.py",
            "nexus run agents:hello -i '{\"name\": \"World\"}'",
            "pytest test_agents.py",
        ],
    }, indent=2)


def main():
    mcp.run()


if __name__ == "__main__":
    main()

# ⚡ Nexus — The Agent Runtime

**Build reliable, composable, observable AI agents with persistent memory and guardrails.**

Nexus is what Docker did for containers — but for AI agents. It's a runtime that makes agents *reliable*.

```python
import nexus
from nexus.context import Context

@nexus.agent(name="summarizer")
async def summarize(text: str, ctx: Context) -> str:
    # Memory — learns from past runs
    past = await ctx.memory.recall("summarizer", "style preferences")

    # LLM — provider-agnostic (Claude, GPT, mock for tests)
    result = await ctx.llm.generate(f"Summarize: {text}")

    # Remember for next time
    await ctx.memory.remember("summarizer", "last_summary", result)
    return result

# Run it
result = await nexus.run(summarize, text="...")
```

## Why Nexus?

Every AI agent today is a **prayer** that the LLM does the right thing. No testing, no memory, no safety, no composition. Nexus fixes all of that:

| Problem | Nexus Solution |
|---------|---------------|
| Agents forget everything | **Persistent Memory** — SQLite-backed, survives restarts |
| Can't test AI code | **Mock LLM** — deterministic tests, no API calls |
| Agents produce unsafe output | **Guardrails** — PII detection, content validation, rate limiting |
| Can't wire agents together | **Composition** — Pipeline, Parallel, Router, Graph (DAG) |
| No visibility into failures | **Tracing** — full execution tree with timing and errors |
| Every run is different | **Structured Output** — Pydantic models, not free text |

## Quick Start

```bash
pip install nexus-agents
nexus init
nexus run agents:hello -i '{"name": "World"}'
```

## Composition

```python
# Sequential pipeline
pipe = nexus.Pipeline(steps=[parser, analyzer, reporter])

# Parallel execution
checks = nexus.Parallel(agents=[linter, tester, reviewer])

# Conditional routing
router = nexus.Router(
    routes={"bug": fixer, "feature": coder, "docs": writer},
    route_fn="auto",  # LLM-based classification
)

# DAG workflow
graph = nexus.Graph(name="ci_pipeline")
graph.add_node("build", builder)
graph.add_node("test", tester, depends_on=["build"])
graph.add_node("lint", linter, depends_on=["build"])
graph.add_node("deploy", deployer, depends_on=["test", "lint"])
```

## Testing

```python
from nexus.testing import AgentTest

async def test_summarizer():
    test = AgentTest(mock_responses=["A concise summary."])
    result = await test.run(summarize, text="Long article...")

    assert result.succeeded
    test.mock.assert_called(times=1)
    test.mock.assert_prompt_contains("Summarize")
```

## Guardrails

```python
from nexus.guardrails import no_pii, content_length, no_code_execution

@nexus.agent(
    name="safe_agent",
    guardrails=[no_pii(), content_length(max_len=5000), no_code_execution()],
)
async def safe_agent(task: str, ctx: Context) -> str:
    return await ctx.llm.generate(task)
    # Output is automatically validated against all guardrails
```

## Claude CLI Integration

```bash
# Nexus works as an MCP server for Claude CLI
nexus mcp

# Or configure in .mcp.json for auto-discovery
```

## License

MIT

# Nexus — The Agent Runtime

## What This Is

Nexus is a Python runtime for building reliable, composable, observable AI agents with persistent memory and guardrails. Think "Docker for AI agents."

## Architecture

```
nexus/
├── agent.py          # @agent decorator + Agent class (core abstraction)
├── context.py        # Execution context (LLM, memory, trace, guardrails, state)
├── runtime.py        # Runtime engine — manages execution lifecycle
├── types.py          # Type system (AgentSpec, AgentResult, Severity, SpanStatus)
├── errors.py         # Exception hierarchy
├── config.py         # Settings from env
├── llm/
│   ├── provider.py   # LLM abstraction (Claude/OpenAI)
│   └── mock.py       # MockLLM for deterministic testing
├── memory/
│   ├── base.py       # MemoryStore interface
│   └── local.py      # SQLite-backed local memory
├── guardrails/
│   ├── engine.py     # Guardrail evaluation
│   └── rules.py      # Built-in rules (no_pii, content_length, etc.)
├── compose/
│   ├── pipeline.py   # Sequential agent pipeline
│   ├── parallel.py   # Concurrent execution
│   ├── router.py     # Conditional dispatch (fn or LLM-based)
│   └── graph.py      # DAG workflow with dependencies
├── observe/
│   └── trace.py      # Execution tracing (Trace + Span tree)
├── testing/
│   └── harness.py    # AgentTest + mock_llm for testing
├── agents/           # Built-in agents (coder, reviewer, researcher)
├── cli.py            # CLI: nexus run, nexus agents, nexus init, nexus mcp
└── mcp/server.py     # MCP server for Claude CLI
```

## Key Commands

```bash
source .venv/bin/activate
pytest tests/ -v                # Run all tests
nexus run MODULE:AGENT -i JSON  # Run an agent
nexus agents                    # List built-in agents
nexus init                      # Scaffold a new project
nexus mcp                       # Start MCP server
```

## Core Pattern

Every agent receives a `Context` with LLM, memory, trace, and guardrails:

```python
@nexus.agent(name="my_agent")
async def my_agent(input: str, ctx: Context) -> str:
    past = await ctx.memory.recall("my_agent", "key")      # Read memory
    result = await ctx.llm.generate("prompt")                # Call LLM
    await ctx.memory.remember("my_agent", "key", result)    # Write memory
    return result                                            # Guardrails check output
```

## Environment

```
NEXUS_AI_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

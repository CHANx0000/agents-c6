# Agents From First Principles — One File, Zero Magic

A single-file, beginners-first AI agent. No classes, no frameworks, no
clever abstractions. Just functions you can read top to bottom in one sitting.

It teaches all the building blocks of an agent:

- **LLM** — the brain (`call_llm`)
- **Tools** — the hands (`TOOLS` dict)
- **RAG** — the library (`rag_lookup`)
- **Memory** — the notebook (`memory` dict)
- **Tracing** — the flight recorder (`trace`)
- **SPAOR loop** — how it all runs together

> **SPAOR** = **S**ense → **P**lan → **A**ct → **O**bserve → **R**eflect
>
> Each letter is one function in `simple_agent.py`. That's the whole agent.

---

## Repository Layout

```
.
├── simple_agent.py              the single-file agent (~250 lines)
├── exercises/                   single-agent exercises (ex1–ex5)
│   ├── ex1_get_time.py
│   ├── ex2_break_search.py
│   ├── ex3_long_term_memory.py
│   ├── ex4_success_condition.py
│   ├── ex5_cost_tracking.py
│   └── TUTORIAL.md              walkthrough for ex1–ex5
├── multiagent/                  multi-agent patterns (ex6–ex8)
│   ├── researcher_agent.py      specialist: collects facts from a KB
│   ├── fact_checker_agent.py    specialist: verifies claims against a KB
│   ├── summarizer_agent.py      specialist: compresses text into one sentence
│   ├── ex6_agent_as_tool.py     agent-as-tool pattern
│   ├── ex7_manager_pattern.py   manager / orchestrator-workers pattern
│   ├── ex8_handoff_pattern.py   handoff / decentralized pattern
│   └── TUTORIAL.md              walkthrough for ex6–ex8
└── requirements.txt
```

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Add your Groq key

```bash
cp .env.example .env
# edit .env and paste your key from https://console.groq.com/keys
```

### 3. Run

```bash
# Single agent
python simple_agent.py

# Multi-agent patterns (run from multiagent/)
cd multiagent
python ex7_manager_pattern.py
python ex8_handoff_pattern.py
```

The agent uses `openai/gpt-oss-120b` on Groq by default.

---

## The SPAOR Loop in Plain English

| Phase | Question it answers | In our code |
|---|---|---|
| **Sense** | "What do I know right now?" | Build a context dict from memory |
| **Plan** | "What should I do next?" | Ask the LLM to choose a tool or COMPLETE |
| **Act** | "Do the thing." | Call the tool function |
| **Observe** | "What actually happened?" | Inspect the tool's result / errors |
| **Reflect** | "Did that help? Keep going?" | Ask the LLM + update memory |

Every "agent loop" you'll ever read (ReAct, OODA, Plan-and-Execute) is a
rearrangement of these same five moves.

---

## Exercises

### Part 1 — Single Agent (`exercises/`)

Do these in order. Each builds on the previous one.

| # | Exercise | Key concept |
|---|---|---|
| 1 | Add a `get_time` tool | Registering tools |
| 2 | Break a tool on purpose | Error handling, hallucination prevention |
| 3 | Add long-term memory | Facts that survive across runs |
| 4 | Enforce a success condition | Hard guards in the loop |
| 5 | Cost tracking | Token budgets per phase |

See `exercises/TUTORIAL.md` for the full walkthrough.

### Part 2 — Multi-Agent Patterns (`multiagent/`)

These introduce three specialists (researcher, fact-checker, summarizer) and
two coordination patterns.

| # | Exercise | Pattern |
|---|---|---|
| 6 | Agent as a tool | One agent calls another like any other tool |
| 7 | Manager pattern | Pure orchestrator whose only tools are agents |
| 8 | Handoff pattern | Triage routes, specialists chain without a manager |

The three specialists:

- **Researcher** — searches a planet KB, collects 2+ facts
- **Fact-checker** — extracts claims, checks each against the KB, produces a VERIFIED/WEAK verdict
- **Summarizer** — compresses text into one sentence

See `multiagent/TUTORIAL.md` for the full walkthrough.

---

## Troubleshooting

**`GROQ_API_KEY not found`** — make sure `.env` exists and contains a real key.

**`JSON parse error`** — the LLM wandered off the format. `parse_json`
already handles this gracefully; if it happens a lot, lower `temperature`
in `call_llm` or sharpen the system prompt.

**Agent loops forever** — the `MAX_ITERATIONS` cap will stop it. Then read
the trace to find which phase made the wrong call.

**Tool errors** — `act` catches tool exceptions and feeds the error string
back into memory, so the agent can react to its own mistakes.

---

## Where to go next

Once you've done the exercises, natural next steps:

- Swap the keyword `rag_lookup` for real embeddings (e.g. `sentence-transformers` + FAISS).
- Replace the text protocol with Groq's native tool-use API.
- Split the single file into modules — only once you *feel* the seams.
- Add more specialists (writer, code reviewer) to the multi-agent patterns.

This repo is the foundation. Everything else in "agent land" — planners,
routers, multi-agent systems, evaluators — is just more of the same five
moves wired together in new ways.

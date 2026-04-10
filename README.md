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
> Act comes before Observe because you can only observe the consequences
> of something that actually happened.

---

## Learning Outcomes

By the end of this lesson you will be able to:

1. Explain the SPAOR loop and point at the line of code that implements each phase.
2. Add a new tool in three lines.
3. Plug in memory so the agent remembers what it tried last turn.
4. Use tracing to debug a run when the agent misbehaves.
5. Define a clear success condition and a max-iteration cap.

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
python simple_agent.py
```

The agent uses `openai/gpt-oss-120b` on Groq by default — it's fast and free to try.

---

## What you'll see

```
============================================================
🎯 GOAL: What is 25 * 4 + 100?
============================================================

--- ITERATION 1 ---

👁  SENSE
  iteration: 1
  goal: What is 25 * 4 + 100?
  history_len: 0
  notes: (empty)

🧠 PLAN
  action: USE_TOOL
  tool: calculate
  args: 25 * 4 + 100
  reasoning: Direct math — use calculate.

⚡ ACT
  tool: calculate
  args: 25 * 4 + 100
  result: 200

🔭 OBSERVE
  kind: result
  ok: True
  summary: 200

💭 REFLECT
  progress: True
  comment: The calculation gives the final value needed.

--- ITERATION 2 ---
...
✅ FINAL ANSWER: 200
```

Every phase prints a line. If something goes wrong, the trace tells you
exactly which phase broke.

---

## The File Tour

`simple_agent.py` is ~250 lines, organized in reading order:

| Section | What it is | Why it matters |
|---|---|---|
| 0. `call_llm` | The only function that touches the API | One choke-point = easy to log, mock, or swap models |
| 1. `TOOLS` | A dict of tool name → function | Tools are just functions. The LLM picks by name. |
| 2. `rag_lookup` | Keyword search over a tiny knowledge base | Real RAG uses embeddings; the shape is the same |
| 3. `memory` | A dict with history + scratchpad | Stateless LLMs need you to hand-feed the past |
| 4. `trace` | Pretty-printer for every phase | The difference between debuggable and mysterious |
| 5. `sense / plan / act / observe / reflect` | Five small functions | **This is the whole agent** |
| 6. `run_agent` | The loop that calls the five functions | 20 lines |
| 7. `parse_json` | Pulls JSON out of an LLM reply | LLMs wrap JSON in prose; handle it once |

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

## Exercises — Progressive

Do these in order. Each builds on the previous one.

### 1. Add a tool

Add `get_time` to `TOOLS`. It should take no useful argument and return the
current time. Run the agent with the goal `"What time is it?"` and confirm
the LLM picks your new tool.

### 2. Break a tool on purpose

Make `tool_search` always return `"no results"`. Run the research goal
again. Does the agent detect the failure and try something else, or does
it hallucinate an answer? Tighten the `reflect` prompt until the agent
gives up cleanly instead of lying.

### 3. Add long-term memory

Extend `memory` with a `facts` list. Every time `reflect` sees
`progress: true`, append a one-line fact to `memory["facts"]`. In `sense`,
pass the last 5 facts into the context. Now run two goals in a row and
notice the second one reuses what the first one learned.

### 4. Enforce a real success condition

Change the goal to:

> "Collect at least 3 distinct facts about planets in the solar system."

Track unique sources in `memory["notes"]` and only allow `COMPLETE` when
there are 3+. The agent should keep searching on its own until the bar is
met, then stop.

### 5. Cost tracking (stretch)

Wrap `call_llm` so every call records `response.usage.total_tokens` into a
`memory["tokens"]` counter. At the end of the run, print the total. Which
phase — Plan or Reflect — burns the most tokens? Why?

### 6. A second agent as a tool (stretch)

Write `tool_summarize(text)` that calls `call_llm` with a "summarize in one
sentence" prompt. Notice that a tool can itself be an LLM call — that's the
entire idea behind "agent-as-tool" and multi-agent systems.

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
- Run two agents in parallel and give one the other as a tool.

This file is the foundation. Everything else in "agent land" — planners,
routers, multi-agent systems, evaluators — is just more of the same five
moves wired together in new ways.

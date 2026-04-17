# Multi-Agent Patterns — Tutorial

This tutorial covers exercises 6-8: the patterns that emerge when you
compose multiple SPAOR agents together. If you have not done exercises 1-5
in `exercises/`, start there — you need to be comfortable with a single
agent's loop before wiring several together.

Each section explains *what changed*, *where it lives in the code*, and *why
it works*.

---

## Directory layout

```
multiagent/
  researcher_agent.py      specialist: collects facts from a KB
  fact_checker_agent.py    specialist: verifies claims against a KB
  summarizer_agent.py      specialist: compresses text into one sentence
  ex6_agent_as_tool.py     main agent + summarizer as a tool
  ex7_manager_pattern.py   manager orchestrating all three specialists
  ex8_handoff_pattern.py   triage + handoff chain between specialists
```

All three specialists follow the same SPAOR shape. Each has private
memory, private tools, and one public entry point (`run_*_agent(text)`)
that takes a string and returns a string.

---

## The three specialists

### `researcher_agent.py`

Given a topic, searches a knowledge base and collects facts.

**Tools:** `kb_search` (keyword match against a planet KB), `list_topics`
(shows available KB keys). Both pure Python.

**Guard:** The loop rejects `COMPLETE` until at least 2 facts are collected.
This prevents the LLM from declaring victory with an empty fact list.

### `fact_checker_agent.py`

Given text with claims, verifies each one against a knowledge base.

**Tools:**
- `extract_claims` — splits text into individual factual assertions
- `check_claim` — looks up one claim against the KB. Filters stop words
  and requires the KB topic key to appear in the claim, preventing false
  matches from generic words like "planet" or "system"
- `verdict` — one LLM call that produces a VERIFIED/WEAK judgment per claim

### `summarizer_agent.py`

Given text, compresses it into one sentence.

**Tools:** `split_text`, `find_topic` (both pure Python), `write_summary`
(one LLM call).

### Why private state matters

Everything outside each agent's public function starts with an underscore
(`_memory`, `_call_llm`, `_TOOLS`). That Python convention for "private"
matters here — it is how we make sure each specialist's state never collides
with anyone else's.

---

## Exercise 6 — A second agent as a tool

**Files:** `summarizer_agent.py` (the specialist) + `ex6_agent_as_tool.py` (the orchestrator)

### What the exercise asks

Write a `tool_summarize` backed by a real second agent — not just a single LLM
call. The specialist agent has its own tools, its own memory, and its own SPAOR
loop. The main agent calls it like any other tool and receives a string back.

### Why a full agent instead of one LLM call?

A bare `call_llm("summarize this")` works, but it is not an agent — it has no
loop, no tools, no ability to check or fix its own work. A real specialist
agent can:

- break the problem into steps (split text → find topic → write summary)
- use tools that do not need LLM calls at all (saving tokens)
- reflect on whether its own output is good enough
- be tested and improved independently of the main agent

The main agent does not care either way. It sends text in and receives a string
back. Swapping a one-liner for a full agent is invisible from the outside.

### Wiring it in

**Step 1 — import the specialist.**

```python
from summarizer_agent import run_summarizer_agent
```

**Step 2 — write `tool_summarize` as a thin wrapper.**

```python
def tool_summarize(text):
    if not text or not text.strip():
        return "nothing to summarize"
    return run_summarizer_agent(text)   # delegates entirely to the specialist
```

That is it. One line delegates to a full SPAOR agent. The main agent never
sees the internal loop — it just receives the returned string.

**Step 3 — register it in `TOOLS`.**

```python
"summarize": {
    "description": (
        "Summarize a block of text in one sentence using a specialist AI agent. "
        "Pass the full text you want summarized. args: text (string)"
    ),
    "function": tool_summarize,
},
```

### The full call tree

```
Main agent (ex6_agent_as_tool.py)
│
│  PLAN: goal is to research and summarize planets
│  ACT:  calls tool_search("jupiter")  →  returns KB text
│  ACT:  calls tool_search("saturn")   →  returns KB text
│  ACT:  calls tool_summarize("Jupiter is... Saturn is...")
│                │
│                └── run_summarizer_agent(text)   [summarizer_agent.py]
│                        │
│                        ├── _tool_split_text(text)   [pure Python]
│                        ├── _tool_find_topic(text)   [pure Python]
│                        └── _tool_write_summary(text) [one LLM call]
│                        │
│                        └── returns "Jupiter and Saturn are..."
│
│  Receives: "Jupiter and Saturn are..."  (just a string, like any tool)
└── COMPLETE with the summary as the final answer
```

### Why separate files instead of one big file?

- **Each agent can be tested independently.** Run `python summarizer_agent.py`
  to check the summarizer works before plugging it into the main agent.
- **Concerns stay separated.** The main agent handles research. The summarizer
  handles compression. Neither bleeds into the other.
- **You can swap the summarizer.** Want a different summarization strategy?
  Change `summarizer_agent.py`. The main agent does not change at all.
- **Private state stays private.** The `_memory` and `_call_llm` in
  `summarizer_agent.py` cannot accidentally overwrite the main agent's memory.

---

## Exercise 7 — The Manager Pattern

**Files:** `researcher_agent.py` + `fact_checker_agent.py` + `summarizer_agent.py` (specialists) + `ex7_manager_pattern.py` (the manager)

### What the exercise asks

Build an outer agent whose *only* tools are other agents. No `search`, no
`calculate`, no `get_time` — just specialists. The outer agent is a pure
orchestrator. Its job is to pick the right specialist, give it the right
input, and combine the outputs into a final answer.

### How it differs from Exercise 6

Exercise 6 added one agent-as-tool (`summarize`) alongside normal tools.
Exercise 7 goes all-in: the manager has **nothing but** agent-tools.

```
ex6 TOOLS = {search, calculate, save_note, get_time, summarize}
                                                      ^-- one agent-tool

ex7 TOOLS = {researcher, fact_checker, summarizer}
              ^            ^              ^
              all three are full SPAOR agents
```

That single change forces the outer agent to *plan the workflow* instead of
reaching for raw tools. It is the same mental step a human team lead makes
when they stop doing the work themselves and start delegating.

### Wrap each specialist as a one-line tool

```python
def tool_researcher(topic):
    return run_researcher_agent(topic)

def tool_fact_checker(text):
    return run_fact_checker_agent(text)

def tool_summarizer(text):
    return run_summarizer_agent(text)

TOOLS = {
    "researcher":   {"description": "...", "function": tool_researcher},
    "fact_checker": {"description": "...", "function": tool_fact_checker},
    "summarizer":   {"description": "...", "function": tool_summarizer},
}
```

### The `LAST_RESEARCH` sentinel

When the manager wants to hand the researcher's output to the fact-checker
or summarizer, it should NOT copy that output into its own prompt — that
would balloon its context window and blow up token usage.

Instead, we teach the manager to write a tiny placeholder:

```json
{"tool": "fact_checker", "args": "LAST_RESEARCH"}
```

And we expand the placeholder in `act()`:

```python
args = decision.get("args") or ""
if args == "LAST_RESEARCH":
    args = memory["last_research"] or ""
```

The long text lives in Python memory, not in the LLM's context. The manager
only has to remember the *word* `LAST_RESEARCH`. That is a 1-token cost vs.
potentially thousands.

This is a reusable trick: whenever your agent needs to pass a large blob
between tool calls, store the blob in code and let the agent reference it
by name.

### The typical workflow

The manager's `plan()` prompt describes the expected workflow:

1. Call **researcher** with a topic → get raw facts.
2. Call **fact_checker** with those facts → get a verified/weak verdict.
3. Call **summarizer** with the verified text → get a one-sentence summary.
4. Declare **COMPLETE** with the summary.

The manager tracks three state flags: `has_research`, `has_verdict`, and
`has_summary`. Its prompt shows the fact-check verdict and the final summary
so the LLM knows exactly what has been done and what remains.

### Tracing a full run

```
--- MANAGER ITERATION 1 ---
🧠 PLAN → action: USE_TOOL, tool: researcher, args: "jupiter and saturn"
  [RESEARCHER AGENT STARTED]
    [RESEARCHER] PLAN → kb_search("jupiter")
    [RESEARCHER] PLAN → kb_search("saturn")
    [RESEARCHER] PLAN → COMPLETE
  [RESEARCHER AGENT DONE]
⚡ ACT → "[jupiter] ... [saturn] ..."

--- MANAGER ITERATION 2 ---
🧠 PLAN → action: USE_TOOL, tool: fact_checker, args: "LAST_RESEARCH"
  [FACT-CHECKER AGENT STARTED]
    [FACT-CHECKER] PLAN → extract_claims(...)
    [FACT-CHECKER] PLAN → check_claim("Jupiter is the largest...")
    [FACT-CHECKER] PLAN → check_claim("Saturn is known for...")
    [FACT-CHECKER] PLAN → verdict(...)
    [FACT-CHECKER] PLAN → COMPLETE
  [FACT-CHECKER AGENT DONE]
⚡ ACT → "Both claims VERIFIED. Status: ALL VERIFIED."

--- MANAGER ITERATION 3 ---
🧠 PLAN → action: USE_TOOL, tool: summarizer, args: "LAST_RESEARCH"
  [SUMMARIZER AGENT STARTED]
    ...full summarizer loop...
  [SUMMARIZER AGENT DONE]
⚡ ACT → "Jupiter and Saturn are..."

--- MANAGER ITERATION 4 ---
🧠 PLAN → action: COMPLETE, answer: "Jupiter and Saturn are..."
```

### The mental model

> The manager is a team lead, not an engineer. It does not open the codebase.
> It picks the right engineer, hands them the ticket, reads the result, and
> decides the next move.

Everything you already know about SPAOR still applies — the only thing that
changed is the *granularity* of a "tool". A tool used to be a function. Now
it is an agent. Zoom out one level and the abstraction is identical.

### When to use this pattern

- You have multiple distinct skills that benefit from their own memory,
  tools, and prompting (research, verification, summarization).
- You want to swap, version, or test each specialist independently.
- You want the outer agent's prompt to stay small — it only needs to know
  *about* specialists, not *how* they work.

If your task fits in one set of tools, don't reach for a manager. This
pattern earns its keep once you have two or more clearly different jobs.

---

## Exercise 8 — The Handoff Pattern

**File:** `ex8_handoff_pattern.py` (reuses all three specialist agents)

### What the exercise asks

Instead of the outer agent *calling* a specialist and waiting for a return
value, the outer agent **transfers control** to a specialist and exits. The
specialist owns the rest of the task. If it needs another specialist, it
hands off again. Control never returns to the original agent.

### Handoff vs. Manager — the one-sentence difference

> Manager is a function call: `result = specialist(input)`.
> Handoff is a `goto`: the specialist becomes the active agent; the caller
> is done.

That sounds small. In practice it changes who decides what happens next.

| | Manager pattern | Handoff pattern |
|---|---|---|
| Who plans the next step? | The manager, after each specialist returns | The currently active specialist |
| Does control come back? | Yes, always | No, never |
| How many specialists run? | Whatever the manager calls in its loop | A chain — each specialist decides if it's terminal or hands off |
| Best when... | You need to *combine* results from multiple specialists | You need to *route* a task to the right specialist once, then get out of the way |

### The triage agent has no loop and no tools

> The triage agent is a single LLM call that outputs a routing decision.
> That is the entire agent.

No SPAOR, no tools, no memory. It does one job: "which specialist should
own this task?" — and then it disappears. It can route to any of the three
specialists: `researcher`, `fact_checker`, or `summarizer`.

```python
def triage(user_goal):
    system = """You are a TRIAGE agent. ... Reply with JSON:
    { "target": "<researcher or fact_checker or summarizer>",
      "handoff_goal": "<rewritten for that specialist>",
      "reasoning": "..." }"""
    raw = call_llm([
        {"role": "system", "content": system},
        {"role": "user", "content": f"User goal: {user_goal}"},
    ])
    return _parse_json(raw)
```

A triage agent being "just a classifier" is not a code smell — it is the
point. Complexity belongs in the specialists, not in the router.

### The handoff contract

Every specialist wrapper returns the same shape:

```python
{"kind": "done",     "answer": "..."}                          # terminal
{"kind": "handoff",  "target": "summarizer", "goal": "..."}    # pass control
```

The wrappers are where the "should I hand off?" policy lives:

```python
def run_researcher_with_handoff(goal, user_goal):
    facts = run_researcher_agent(goal)
    if _wants_verification(user_goal) or _wants_summary(user_goal):
        return {"kind": "handoff", "target": "fact_checker", "goal": facts}
    return {"kind": "done", "answer": facts}


def run_fact_checker_with_handoff(goal, user_goal):
    verdict = run_fact_checker_agent(goal)
    if "WEAK" in verdict.upper() and not _already_retried(user_goal):
        return {"kind": "handoff", "target": "researcher", "goal": goal}
    if _wants_summary(user_goal):
        combined = f"{goal}\n\nFact-check verdict: {verdict}"
        return {"kind": "handoff", "target": "summarizer", "goal": combined}
    return {"kind": "done", "answer": verdict}


def run_summarizer_with_handoff(goal, user_goal):
    return {"kind": "done", "answer": run_summarizer_agent(goal)}
```

Two things worth noting:

1. The **fact-checker can hand back to the researcher** if it finds weak
   claims. This is the tight feedback loop — the fact-checker and researcher
   can iterate without involving triage.
2. A **retry guard** (`_already_retried`) prevents infinite ping-pong. The
   fact-checker gets one chance to ask for more evidence. After that, it
   chains forward to the summarizer regardless.

### The orchestrator is a while loop

```python
def run_with_handoff(user_goal):
    decision = triage(user_goal)
    current_target = decision["target"]
    current_goal = decision["handoff_goal"]

    for hop in range(MAX_HANDOFFS):
        result = AGENTS[current_target](current_goal, user_goal)
        if result["kind"] == "done":
            return result["answer"]
        current_target = result["target"]
        current_goal = result["goal"]
```

There is no outer "brain" making decisions each hop — the loop just does
what the *current* specialist says. This is literally a `goto` between
agents, implemented in ten lines.

### Tracing a full run

```
🎯 USER GOAL: Research Jupiter and Saturn, verify the facts, and give me a one-sentence summary.

[TRIAGE] → researcher
  [RESEARCHER] ...SPAOR loop... → 2 facts

[HANDOFF] researcher → fact_checker
  [FACT-CHECKER] ...extracts claims, checks each one... → Both claims VERIFIED

[HANDOFF] fact_checker → summarizer
  [SUMMARIZER] ...SPAOR loop... → Jupiter and Saturn are...

FINAL ANSWER (from summarizer): Jupiter and Saturn are the two largest...
Handoffs used: 3
```

### The mental model

> **Handoff** is the receptionist pattern. You walk into a building, say what
> you need, the receptionist points you down the hall, and you go. The
> receptionist does not follow you. If the person you reach realizes you
> actually need someone else, *they* walk you over. You never return to
> the front desk.
>
> **Manager** is the general contractor pattern. You tell the contractor what
> you want. They call the plumber, wait for the plumber to finish, call the
> electrician, wait, and finally tell you the job is done. You talk to the
> contractor the whole time.

Both are useful. Neither is strictly better.

### When to use handoffs

- Triage / intake flows where the right specialist depends on the input.
- Multi-stage pipelines where each stage naturally owns what happens next.
- Situations where you want the active agent's prompt to stay focused on
  its own job — it does not need to know about the orchestrator at all.

### When NOT to use handoffs

- If you need to *combine* outputs from several specialists into one answer.
  That is what a manager is for — the manager is the only one with the full
  picture. A handoff chain cannot look backwards.
- If you need the original caller to enforce success conditions across the
  chain. Once triage hands off, it has no way to reject the final answer.

---

## Putting it all together

| Concept | Where it lives | Plain English |
|---|---|---|
| Agent-as-tool | `ex6` — a tool function that calls `run_summarizer_agent()` | A full SPAOR agent wrapped as a tool; the caller sees only a string |
| Fact-checker | `fact_checker_agent.py` — extract/check/verdict tools | A specialist that verifies claims against a KB before they are trusted |
| Manager pattern | `ex7` — `TOOLS` dict contains only other agents | A team lead: picks specialists (researcher → fact_checker → summarizer), combines their output, owns the final answer |
| Handoff pattern | `ex8` — triage transfers control, specialists chain | A receptionist: points you to the right person and disappears; specialists can hand off to each other (including backwards for re-research) |

The single mental model: **an agent is just a Python `for` loop, and the
LLM is just a function that reads text and writes text**. Multi-agent is
the same loop, where the "tools" happen to be other loops.

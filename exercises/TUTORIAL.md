# How the Exercises Were Solved — A Beginner's Tutorial

This tutorial walks through every exercise in plain language. You do not need
to be an expert. If you have read `simple_agent.py` and understood the basic
SPAOR loop (Sense → Plan → Act → Observe → Reflect), you are ready.

Each section explains *what changed*, *where it lives in the code*, and *why
it works*.

---

## Background: how the agent decides what to do

Before diving into the exercises it helps to be clear on one thing: **the
agent does not magically know which tool to call**. It knows because we write
the tool's name and description into the prompt we send to the LLM. The LLM
reads that list and picks the best match for the goal.

```
Available tools:
- search: Search the knowledge base for info about a topic. args: topic (string)
- calculate: Evaluate a math expression. args: expression (string)
```

That is the entire secret. Adding a tool means adding its name and description
to that list. Nothing else changes in the loop.

---

## Exercise 1 — Add a `get_time` tool

**File:** `ex1_get_time.py`

### What the exercise asks

Add a tool that returns the current time and verify the LLM picks it when the
goal is "What time is it?".

### What we changed

**Step 1 — write the function.**

```python
from datetime import datetime

def tool_get_time(_ignored=None):
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

`datetime.now()` gives us the current local time. `.strftime(...)` formats it
as a human-readable string. The parameter is called `_ignored` because the
agent always passes *some* argument to every tool (it is how the loop is
built), but we do not need one here.

**Step 2 — register it in `TOOLS`.**

```python
TOOLS = {
    ...
    "get_time": {
        "description": "Get the current local time. Pass an empty string as the argument.",
        "function": tool_get_time,
    },
}
```

That is literally all. The next time `plan()` builds the tool list string and
sends it to the LLM, `get_time` appears in the menu. When the goal is "What
time is it?", the LLM matches those words to the description and picks it.

### The mental model

Think of `TOOLS` as a restaurant menu. The LLM is the waiter who reads the
menu and takes your order. You do not need to change the waiter — you just add
a new item to the menu.

---

## Exercise 2 — Break a tool on purpose

**File:** `ex2_break_search.py`

### What the exercise asks

Force `tool_search` to always return `"no results"`. Watch whether the agent
hallucinates an answer or admits it cannot find anything. Tighten the `reflect`
prompt until the agent gives up cleanly.

### What we changed

**Step 1 — add a flag to break the tool.**

```python
BREAK_SEARCH = True

def tool_search(topic):
    if BREAK_SEARCH:
        return "no results"
    return rag_lookup(topic)
```

One `if` statement is enough. Flip the flag to `False` to restore normal
behaviour.

**Step 2 — teach `observe` that "no results" is a failure.**

In the original code, `observe` only marked a result as bad if it started with
`"ERROR"`. But `"no results"` is not an error message — it is a silent failure.
The agent would see `ok=True` and assume things went fine.

```python
elif isinstance(result, str) and result.strip() == "no results":
    observation = {"kind": "empty", "ok": False, "summary": "search returned no results"}
```

Now `ok=False` flows into `reflect`.

**Step 3 — tighten the `reflect` prompt.**

Before this exercise, `reflect` could write whatever it liked. That left a
gap for the LLM to say "yes, progress was made" even when the tool returned
nothing. We closed that gap with explicit rules:

```
Rules you MUST follow:
1. If observation ok is false, progress MUST be false.
2. Never invent or guess information that the tools did not return.
3. If the same tool keeps failing, recommend giving up cleanly.
```

We also updated the `plan` system prompt to say: if every tool has failed,
set `action` to `COMPLETE` and honestly admit the failure.

### Why hallucination happens (and how rules stop it)

LLMs are trained to be helpful. "Helpful" to the model sometimes means
producing a confident-sounding answer even when the evidence is missing. The
fix is not a better model — it is a clearer contract in the prompt. When you
write *"if observation ok is false, progress MUST be false"*, you are giving
the model a rule it can follow instead of guessing.

---

## Exercise 3 — Add long-term memory (`facts`)

**File:** `ex3_long_term_memory.py`

### What the exercise asks

Add a `facts` list to `memory`. Every time `reflect` sees `progress: true`,
store a one-line fact. Pass the last 5 facts into `sense` so the next
iteration (and the next `run_agent()` call) can reuse them.

### What we changed

**Step 1 — extend `memory`.**

```python
memory = {
    "history": [],
    "notes": [],
    "facts": [],   # <-- new
}
```

**Step 2 — preserve facts across runs.**

The original `reset_memory()` cleared everything. We renamed it
`reset_run_state()` and deliberately left `facts` alone:

```python
def reset_run_state():
    memory["history"].clear()
    memory["notes"].clear()
    # memory["facts"] is NOT cleared — it survives between runs
```

**Step 3 — ask `reflect` to extract a fact.**

We added a `"fact"` field to the JSON the LLM returns:

```python
prompt = """...
Reply with JSON only:
{
  "progress": true or false,
  "fact": "<one short factual sentence learned from the observation, or empty string>",
  "comment": "<one sentence>"
}"""
```

Then we read it back and append it to the list:

```python
if judgment.get("progress") and judgment.get("fact"):
    fact = judgment["fact"].strip()
    if fact and fact not in memory["facts"]:
        memory["facts"].append(fact)
```

The `not in` check avoids storing the same fact twice.

**Step 4 — inject facts into `sense`.**

```python
recent_facts = memory["facts"][-5:]
context = {
    ...
    "facts": recent_facts,
}
```

And in the `plan` prompt:

```
Known facts from previous runs: {context['facts']}
```

Now when you call `run_agent()` a second time, the agent reads last run's
facts before deciding what to do. It can skip re-searching things it already
knows.

### The mental model

`history` is short-term memory (last few steps, reset each run).
`facts` is long-term memory (survives forever until you clear it manually).
Humans work the same way.

---

## Exercise 4 — Enforce a real success condition

**File:** `ex4_success_condition.py`

### What the exercise asks

Goal: collect at least 3 distinct facts about planets. Track unique sources.
Only allow `COMPLETE` when 3+ sources have been found.

### What we changed

**Step 1 — track unique sources.**

Every time `tool_search` returns real results, we parse the `[source]` tag from
each line and record it:

```python
def tool_search(topic):
    result = rag_lookup(topic)
    if result != "no results":
        for line in result.splitlines():
            m = re.match(r"\[([^\]]+)\]", line)
            if m:
                source = m.group(1)
                if source not in memory["sources"]:
                    memory["sources"].append(source)
    return result
```

**Step 2 — add `sources` to memory.**

```python
memory = {
    ...
    "sources": [],
}
```

**Step 3 — gate `COMPLETE` on the real bar.**

Inside the loop, after the agent declares `COMPLETE`, we check:

```python
if decision.get("action") == "COMPLETE":
    if len(memory["sources"]) < REQUIRED_SOURCES:
        print("Agent tried to COMPLETE too early. Forcing continuation.")
        memory["history"].append({
            "tool": None,
            "args": None,
            "observation": f"COMPLETE rejected: only {len(memory['sources'])} sources",
            "progress": False,
        })
        continue   # go back to the top of the loop
```

`continue` skips `break` and restarts the iteration. The agent sees the
rejection in its history on the next `sense`, understands it needs more
sources, and keeps searching.

**Step 4 — tell the agent about the bar upfront.**

We also added this to the `plan` system prompt:

```
IMPORTANT: Do NOT set action to COMPLETE until you have found at least 3
distinct planet sources. Keep searching different planets.
```

And we pass the current count into the user message:

```
Sources found so far: ['mars', 'jupiter'] (2/3 needed)
```

This gives the agent the information it needs to self-regulate, which means
we should rarely need the hard gate in step 3. But the gate is the safety net.

### Why both the prompt hint and the hard gate?

Prompts alone can fail — the LLM can miscount or get impatient. The hard gate
in the loop is a guarantee that the code enforces, not the model. Always put
critical constraints in code, not just in prompts.

---

## Exercise 5 — Cost tracking

**File:** `ex5_cost_tracking.py`

### What the exercise asks

Record `response.usage.total_tokens` on every LLM call. At the end, print
which phase (Plan or Reflect) burned the most tokens and why.

### What we changed

**Step 1 — add a `phase` parameter to `call_llm`.**

```python
def call_llm(messages, phase="other"):
    response = client.chat.completions.create(...)
    usage = getattr(response, "usage", None)
    if usage:
        total = getattr(usage, "total_tokens", 0) or 0
        memory["tokens"]["total"] += total
        memory["tokens"][phase] = memory["tokens"].get(phase, 0) + total
    return response.choices[0].message.content
```

`getattr(..., None)` is safe — if the API ever stops returning usage data, the
code does not crash.

**Step 2 — label every call site.**

```python
# in plan():
raw = call_llm([...], phase="plan")

# in reflect():
raw = call_llm([...], phase="reflect")
```

**Step 3 — add `tokens` to memory.**

```python
memory = {
    ...
    "tokens": {"total": 0},
}
```

Reset it at the start of each run with:

```python
memory["tokens"] = {"total": 0}
```

**Step 4 — print the report.**

```python
def print_token_report():
    tokens = memory["tokens"]
    total = tokens.get("total", 0)
    phases = {k: v for k, v in tokens.items() if k != "total"}
    for phase, count in sorted(phases.items(), key=lambda x: -x[1]):
        pct = (count / total * 100) if total else 0
        bar = "█" * int(pct / 5)
        print(f"   {phase:12s}: {count:6d}  ({pct:5.1f}%)  {bar}")
```

### Which phase burns more, and why?

`plan` typically burns more tokens than `reflect`. Here is why:

- `plan` receives the full tool list, the entire recent history, all notes, all
  known facts, and the goal. Its input is large.
- `reflect` receives only the last action and its observation — a much smaller
  slice of context.

The more context you send, the more tokens you pay for. This is a real
consideration when building production agents: keep prompts lean, pass only
what the current phase actually needs.

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

### Part 1 — The specialist: `summarizer_agent.py`

The summarizer agent is built exactly like `simple_agent.py`. It just has a
different set of tools and a narrower goal.

**Its three tools:**

```python
def _tool_split_text(text):
    """Split the input into numbered sentences. Pure Python — no LLM."""
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in raw if s.strip()]
    lines = [f"{i + 1}. {s}" for i, s in enumerate(sentences)]
    return f"{len(sentences)} sentence(s) found:\n" + "\n".join(lines)


def _tool_find_topic(text):
    """Find the most frequently mentioned meaningful words. Pure Python."""
    stop_words = {"the", "a", "an", "is", "it", "in", ...}
    words = re.findall(r"[a-z]+", text.lower())
    meaningful = [w for w in words if w not in stop_words and len(w) > 3]
    most_common = Counter(meaningful).most_common(3)
    return f"Top topic words: {', '.join(w for w, _ in most_common)}"


def _tool_write_summary(text):
    """Ask the LLM to write a one-sentence summary. Only LLM call."""
    reply = _call_llm([
        {"role": "system", "content": "Write ONE clear sentence summarizing the text."},
        {"role": "user", "content": text},
    ], phase="write_summary")
    return reply.strip()
```

Notice that two of the three tools are pure Python — no LLM needed. The agent
uses the LLM only where human-like language understanding is actually required.

**Its public interface:**

```python
def run_summarizer_agent(text):
    """Run the summarizer's SPAOR loop. Returns a one-sentence summary string."""
    _memory["history"].clear()
    summary = None
    for i in range(1, MAX_ITERATIONS + 1):
        context = _sense(text, i)
        decision = _plan(context)
        if decision.get("action") == "COMPLETE":
            summary = decision.get("answer")
            break
        result = _act(decision)
        observation = _observe(decision, result)
        _reflect(text, decision, observation)
    return summary or "(no summary produced)"
```

This is the only function the outside world needs to call. Everything else
(`_tools`, `_memory`, `_call_llm`) is private (prefixed with `_`) so it
cannot accidentally collide with the main agent's state.

**Typical run inside the summarizer:**

```
[SUMMARIZER] SENSE  → goal: "Jupiter is the largest planet..."
[SUMMARIZER] PLAN   → action: USE_TOOL, tool: split_text
[SUMMARIZER] ACT    → "2 sentence(s) found: 1. Jupiter... 2. Saturn..."
[SUMMARIZER] REFLECT→ progress: true

[SUMMARIZER] PLAN   → action: USE_TOOL, tool: find_topic
[SUMMARIZER] ACT    → "Top topic words: jupiter, planet, storm"
[SUMMARIZER] REFLECT→ progress: true

[SUMMARIZER] PLAN   → action: USE_TOOL, tool: write_summary
[SUMMARIZER] ACT    → "Jupiter and Saturn are the solar system's most..."
[SUMMARIZER] REFLECT→ progress: true

[SUMMARIZER] PLAN   → action: COMPLETE, answer: "Jupiter and Saturn..."
```

### Part 2 — Wiring it into the main agent: `ex6_agent_as_tool.py`

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

## Putting it all together

Here is a summary of the single mental model that connects every exercise:

| Concept | Where it lives | Plain English |
|---|---|---|
| Tool | A Python function + entry in `TOOLS` dict | Something the agent can *do* |
| Tool description | The `"description"` string in `TOOLS` | What the LLM reads to decide which tool to call |
| Observation | Output of `observe()` | What actually happened after the tool ran |
| Reflection | Output of `reflect()` | Did that help? What did we learn? |
| Short-term memory | `memory["history"]` | The last few steps, reset each run |
| Long-term memory | `memory["facts"]` | Lessons kept across multiple runs |
| Success condition | A check in the loop before accepting `COMPLETE` | Code-enforced guarantee that the goal is really met |
| Cost | `response.usage.total_tokens` | How much context you sent + received |
| Agent-as-tool | A tool function that calls `run_summarizer_agent()` | A full SPAOR agent wrapped as a tool; the caller sees only a string |

Every exercise is really the same lesson restated: **the agent loop is just a
Python `for` loop, and the LLM is just a function that reads text and writes
text**. All the "intelligence" comes from the descriptions you write and the
rules you enforce in ordinary Python code.

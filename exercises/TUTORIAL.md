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

## Next steps

Exercises 6-8 introduce multi-agent patterns (agent-as-tool, manager,
handoff). They live in the `multiagent/` directory with their own
`TUTORIAL.md`. Head there once you are comfortable with the single-agent
concepts above.

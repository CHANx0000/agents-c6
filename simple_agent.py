"""
simple_agent.py — A beginner-friendly AI agent in one file.

Everything you need to understand agents from first principles lives here:
    1. An LLM call                     (the brain)
    2. A tool registry                 (the hands)
    3. Short-term memory               (the notebook)
    4. A tiny RAG "knowledge base"     (the library)
    5. A trace log                     (the flight recorder)
    6. The SPAOR loop that ties it all together

SPAOR = Sense → Plan → Act → Observe → Reflect

Run it:
    pip install groq python-dotenv
    echo 'GROQ_API_KEY=gsk-...' > .env
    python simple_agent.py

No classes. Only functions. Read top to bottom.
"""

import json
import os
import re

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ---------------------------------------------------------------------------
# 0. The LLM
# ---------------------------------------------------------------------------
# One function, one job: take a list of messages, return a string.
# Every other piece of the agent calls this — nothing else talks to the API.

MODEL = "openai/gpt-oss-120b"
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def call_llm(messages):
    """Send messages to Groq and return the assistant's text reply."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,  # low temp = more predictable tool calls
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# 1. Tools  (the agent's "hands")
# ---------------------------------------------------------------------------
# A tool is a name, a description, and a function. The LLM reads the
# descriptions and decides which one to call.

def tool_search(topic):
    """Pretend-search our tiny knowledge base (see section 2)."""
    return rag_lookup(topic)


def tool_calculate(expression):
    """Evaluate a math expression like '25 * 4 + 100'."""
    # eval is fine here because this is a toy. Never do this in production.
    return str(eval(expression))


def tool_save_note(text):
    """Write a note to the agent's scratchpad (see section 3)."""
    memory["notes"].append(text)
    return f"saved note ({len(memory['notes'])} total)"


TOOLS = {
    "search": {
        "description": "Search the knowledge base for info about a topic. args: topic (string)",
        "function": tool_search,
        "arg_name": "topic",
    },
    "calculate": {
        "description": "Evaluate a math expression. args: expression (string)",
        "function": tool_calculate,
        "arg_name": "expression",
    },
    "save_note": {
        "description": "Save a short note to scratchpad for later. args: text (string)",
        "function": tool_save_note,
        "arg_name": "text",
    },
}


# ---------------------------------------------------------------------------
# 2. Mini RAG  (the agent's "library")
# ---------------------------------------------------------------------------
# Real RAG uses embeddings + a vector DB. The *idea* is simpler: given a
# query, return the most relevant chunks of text. We fake that with keyword
# overlap so you can see the shape without any extra dependencies.

KNOWLEDGE_BASE = {
    "mars": "Mars is the fourth planet from the Sun. NASA's Perseverance rover landed in Jezero Crater in February 2021.",
    "jupiter": "Jupiter is the largest planet in the solar system, a gas giant famous for the Great Red Spot storm.",
    "saturn": "Saturn is known for its extensive ring system of ice and rock, and has 146 known moons including Titan.",
    "python": "Python is a high-level programming language known for readability. Created by Guido van Rossum in 1991.",
    "groq": "Groq builds fast inference hardware and hosts open-weight LLMs via a cloud API.",
}


def rag_lookup(query):
    """Return KB entries whose key or text overlaps with the query words."""
    words = set(re.findall(r"[a-z0-9]+", query.lower()))
    hits = []
    for key, text in KNOWLEDGE_BASE.items():
        key_words = set(re.findall(r"[a-z0-9]+", (key + " " + text).lower()))
        if words & key_words:
            hits.append(f"[{key}] {text}")
    return "\n".join(hits) if hits else "no results"


# ---------------------------------------------------------------------------
# 3. Memory  (the agent's "notebook")
# ---------------------------------------------------------------------------
# One plain dict. `history` is the chronological list of what happened each
# iteration; `notes` is a scratchpad the agent can write to via save_note.

memory = {
    "history": [],  # list of {iteration, plan, result, reflection}
    "notes": [],    # scratchpad strings
}


def reset_memory():
    memory["history"].clear()
    memory["notes"].clear()


# ---------------------------------------------------------------------------
# 4. Tracing  (the agent's "flight recorder")
# ---------------------------------------------------------------------------
# Print every SPAOR phase with a consistent emoji header so you can read the
# run like a story. In a real system you would also append to a JSONL file.

def trace(phase, data):
    print(f"\n{phase}")
    for key, value in data.items():
        preview = str(value)
        if len(preview) > 120:
            preview = preview[:117] + "..."
        print(f"  {key}: {preview}")


# ---------------------------------------------------------------------------
# 5. SPAOR phases  (five small functions — this is the whole agent)
# ---------------------------------------------------------------------------
def sense(goal, iteration):
    """SENSE: collect everything the agent currently knows."""
    context = {
        "iteration": iteration,
        "goal": goal,
        "history": memory["history"][-3:],  # last 3 steps, keeps prompt small
        "notes": memory["notes"],
    }
    trace("👁  SENSE", {
        "iteration": iteration,
        "goal": goal,
        "history_len": len(memory["history"]),
        "notes": memory["notes"] or "(empty)",
    })
    return context


def plan(context):
    """PLAN: ask the LLM to pick a tool (or say it is done)."""
    tool_list = "\n".join(
        f"- {name}: {info['description']}" for name, info in TOOLS.items()
    )

    system = f"""You are a careful research agent solving a goal step by step.

Available tools:
{tool_list}

On every turn reply with ONE JSON object and nothing else:
{{
  "action": "USE_TOOL" or "COMPLETE",
  "tool": "<tool name or null>",
  "args": "<single string argument or null>",
  "reasoning": "<why you chose this>",
  "answer": "<final answer if COMPLETE, else null>"
}}"""

    user = f"""Goal: {context['goal']}
Iteration: {context['iteration']}
Recent history: {json.dumps(context['history'], default=str)}
Notes: {context['notes']}

Decide the next action."""

    raw = call_llm([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])
    decision = parse_json(raw)

    trace("🧠 PLAN", {
        "action": decision.get("action"),
        "tool": decision.get("tool"),
        "args": decision.get("args"),
        "reasoning": decision.get("reasoning"),
    })
    return decision


def act(decision):
    """ACT: actually run the tool the plan chose."""
    if decision.get("action") != "USE_TOOL":
        trace("⚡ ACT", {"result": "no tool — agent chose COMPLETE"})
        return None

    tool_name = decision.get("tool")
    if tool_name not in TOOLS:
        result = f"ERROR: unknown tool '{tool_name}'"
        trace("⚡ ACT", {"tool": tool_name, "result": result})
        return result

    try:
        result = TOOLS[tool_name]["function"](decision.get("args"))
    except Exception as e:  # keep the loop alive on tool failure
        result = f"ERROR: {e}"

    trace("⚡ ACT", {"tool": tool_name, "args": decision.get("args"), "result": result})
    return result


def observe(decision, result):
    """OBSERVE: look at what actually happened after Act.

    This is the ReAct-style 'observation' step: we inspect the tool's
    output and turn it into a structured snapshot the Reflect phase can
    reason about. No LLM call — just plain inspection.
    """
    if decision.get("action") != "USE_TOOL":
        observation = {"kind": "none", "ok": True, "summary": "no action taken"}
    elif isinstance(result, str) and result.startswith("ERROR"):
        observation = {"kind": "error", "ok": False, "summary": result}
    else:
        preview = str(result)
        if len(preview) > 200:
            preview = preview[:197] + "..."
        observation = {"kind": "result", "ok": True, "summary": preview}

    trace("🔭 OBSERVE", observation)
    return observation


def reflect(goal, decision, observation):
    """REFLECT: ask the LLM whether the observation gets us closer to the goal."""
    if decision.get("action") == "COMPLETE":
        trace("💭 REFLECT", {"done": True, "reason": "agent declared COMPLETE"})
        return True  # done

    prompt = f"""Goal: {goal}
Last action: {decision.get('tool')}({decision.get('args')})
Observation: {observation['summary']}
Observation ok: {observation['ok']}

In ONE short sentence: did this get us closer to the goal? Reply with JSON:
{{"progress": true or false, "comment": "<one sentence>"}}"""

    raw = call_llm([
        {"role": "system", "content": "You judge agent progress. Reply with JSON only."},
        {"role": "user", "content": prompt},
    ])
    judgment = parse_json(raw)
    trace("💭 REFLECT", judgment)

    # Record what happened so the next SENSE phase can see it.
    memory["history"].append({
        "tool": decision.get("tool"),
        "args": decision.get("args"),
        "observation": observation["summary"],
        "progress": judgment.get("progress"),
    })
    return False  # not done yet


# ---------------------------------------------------------------------------
# 6. The SPAOR loop itself
# ---------------------------------------------------------------------------
MAX_ITERATIONS = 6


def run_agent(goal):
    """Run SPAOR until the agent says it's done or we hit the cap."""
    reset_memory()
    print("=" * 60)
    print(f"🎯 GOAL: {goal}")
    print("=" * 60)

    final_answer = None

    for i in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- ITERATION {i} ---")

        # S — what do we know?
        context = sense(goal, i)

        # P — what should we do next?
        decision = plan(context)

        # If the agent says it's done, skip A/O and wrap up.
        if decision.get("action") == "COMPLETE":
            final_answer = decision.get("answer") or "(no answer provided)"
            reflect(goal, decision, {"kind": "none", "ok": True, "summary": ""})
            break

        # A — do the thing
        result = act(decision)

        # O — look at what happened
        observation = observe(decision, result)

        # R — did that help? Update memory + decide whether to loop.
        done = reflect(goal, decision, observation)
        if done:
            break
    else:
        print(f"\n⚠️  Hit max iterations ({MAX_ITERATIONS})")

    print("\n" + "=" * 60)
    print(f"FINAL ANSWER: {final_answer or '(none)'}")
    print("=" * 60)
    return final_answer


# ---------------------------------------------------------------------------
# 7. Helpers
# ---------------------------------------------------------------------------
def parse_json(text):
    """Extract the first {...} JSON object from an LLM reply.

    LLMs sometimes wrap JSON in prose or markdown fences. We grab the first
    balanced { ... } block and try to parse it. If that fails, we return a
    safe default so the loop can continue instead of crashing.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"action": "COMPLETE", "answer": text.strip()}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"action": "COMPLETE", "answer": text.strip()}


# ---------------------------------------------------------------------------
# 8. Try it
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Swap these out, add your own, or call run_agent() from a REPL.
    run_agent("What is 25 * 4 + 100?")
    # run_agent("Tell me what you know about Mars and Jupiter.")
    # run_agent("Calculate 100 / 4, then look up info about that number.")

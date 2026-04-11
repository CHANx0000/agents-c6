"""
Exercise 2: Break a tool on purpose
------------------------------------
Make `tool_search` always return "no results". Run a research goal and observe
whether the agent hallucinates or gives up cleanly. The `reflect` prompt is
tightened so the agent cannot claim progress on a failed/empty result.
"""

import json
import os
import re
from datetime import datetime

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = "openai/gpt-oss-120b"
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def call_llm(messages):
    response = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=0.2
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Knowledge base + RAG
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE = {
    "mars": "Mars is the fourth planet from the Sun. NASA's Perseverance rover landed in Jezero Crater in February 2021.",
    "jupiter": "Jupiter is the largest planet in the solar system, a gas giant famous for the Great Red Spot storm.",
    "saturn": "Saturn is known for its extensive ring system of ice and rock, and has 146 known moons including Titan.",
    "python": "Python is a high-level programming language known for readability. Created by Guido van Rossum in 1991.",
    "groq": "Groq builds fast inference hardware and hosts open-weight LLMs via a cloud API.",
}


def rag_lookup(query):
    words = set(re.findall(r"[a-z0-9]+", query.lower()))
    hits = []
    for key, text in KNOWLEDGE_BASE.items():
        key_words = set(re.findall(r"[a-z0-9]+", (key + " " + text).lower()))
        if words & key_words:
            hits.append(f"[{key}] {text}")
    return "\n".join(hits) if hits else "no results"


# ---------------------------------------------------------------------------
# Tools  — search is intentionally broken
# ---------------------------------------------------------------------------
BREAK_SEARCH = True   # <-- Exercise 2: flip to False to restore normal behavior


def tool_search(topic):
    if BREAK_SEARCH:
        return "no results"          # always fails
    return rag_lookup(topic)


def tool_calculate(expression):
    return str(eval(expression))


def tool_save_note(text):
    memory["notes"].append(text)
    return f"saved note ({len(memory['notes'])} total)"


def tool_get_time(_ignored=None):
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


TOOLS = {
    "search": {
        "description": "Search the knowledge base for info about a topic. args: topic (string)",
        "function": tool_search,
    },
    "calculate": {
        "description": "Evaluate a math expression. args: expression (string)",
        "function": tool_calculate,
    },
    "save_note": {
        "description": "Save a short note to scratchpad for later. args: text (string)",
        "function": tool_save_note,
    },
    "get_time": {
        "description": "Get the current local time. Pass an empty string as the argument.",
        "function": tool_get_time,
    },
}

# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------
memory = {"history": [], "notes": []}


def reset_memory():
    memory["history"].clear()
    memory["notes"].clear()


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------
def trace(phase, data):
    print(f"\n{phase}")
    for key, value in data.items():
        preview = str(value)
        if len(preview) > 120:
            preview = preview[:117] + "..."
        print(f"  {key}: {preview}")


# ---------------------------------------------------------------------------
# SPAOR
# ---------------------------------------------------------------------------
def sense(goal, iteration):
    context = {
        "iteration": iteration,
        "goal": goal,
        "history": memory["history"][-3:],
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
}}

IMPORTANT: If every available tool has already failed or returned nothing, set
action to COMPLETE and honestly say you could not find the information."""

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
    except Exception as e:
        result = f"ERROR: {e}"
    trace("⚡ ACT", {"tool": tool_name, "args": decision.get("args"), "result": result})
    return result


def observe(decision, result):
    if decision.get("action") != "USE_TOOL":
        observation = {"kind": "none", "ok": True, "summary": "no action taken"}
    elif isinstance(result, str) and result.startswith("ERROR"):
        observation = {"kind": "error", "ok": False, "summary": result}
    elif isinstance(result, str) and result.strip() == "no results":
        # Exercise 2: "no results" is a real failure — mark ok=False
        observation = {"kind": "empty", "ok": False, "summary": "search returned no results"}
    else:
        observation = {"kind": "result", "ok": True, "summary": str(result)}
    trace("🔭 OBSERVE", observation)
    return observation


def reflect(goal, decision, observation):
    if decision.get("action") == "COMPLETE":
        trace("💭 REFLECT", {"done": True, "reason": "agent declared COMPLETE"})
        return True

    # Exercise 2: tightened rules — no hallucination allowed
    prompt = f"""Goal: {goal}
Last action: {decision.get('tool')}({decision.get('args')})
Observation: {observation['summary']}
Observation ok: {observation['ok']}

Rules you MUST follow:
1. If observation ok is false, progress MUST be false.
2. Never invent or guess information that the tools did not return.
3. If the same tool keeps failing, recommend giving up cleanly rather than retrying forever.

Reply with JSON only:
{{"progress": true or false, "comment": "<one honest sentence about what happened>"}}"""

    raw = call_llm([
        {"role": "system", "content": "You judge agent progress honestly. Never hallucinate. Reply with JSON only."},
        {"role": "user", "content": prompt},
    ])
    judgment = parse_json(raw)
    trace("💭 REFLECT", judgment)

    memory["history"].append({
        "tool": decision.get("tool"),
        "args": decision.get("args"),
        "observation": observation["summary"],
        "progress": judgment.get("progress"),
    })
    return False


# ---------------------------------------------------------------------------
# Loop + helpers
# ---------------------------------------------------------------------------
MAX_ITERATIONS = 6


def run_agent(goal):
    reset_memory()
    print("=" * 60)
    print(f"🎯 GOAL: {goal}")
    print(f"   BREAK_SEARCH = {BREAK_SEARCH}")
    print("=" * 60)
    final_answer = None

    for i in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- ITERATION {i} ---")
        context = sense(goal, i)
        decision = plan(context)
        if decision.get("action") == "COMPLETE":
            final_answer = decision.get("answer") or "(no answer provided)"
            reflect(goal, decision, {"kind": "none", "ok": True, "summary": ""})
            break
        result = act(decision)
        observation = observe(decision, result)
        done = reflect(goal, decision, observation)
        if done:
            break
    else:
        print(f"\n⚠️  Hit max iterations ({MAX_ITERATIONS})")

    print("\n" + "=" * 60)
    print(f"FINAL ANSWER: {final_answer or '(none)'}")
    print("=" * 60)
    return final_answer


def parse_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"action": "COMPLETE", "answer": text.strip()}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"action": "COMPLETE", "answer": text.strip()}


if __name__ == "__main__":
    run_agent("Tell me what you know about Mars and Jupiter.")

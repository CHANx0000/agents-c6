"""
Exercise 5: Cost tracking (stretch)
-------------------------------------
Wrap `call_llm` so every call records response.usage.total_tokens into a
memory["tokens"] counter, broken down by phase (plan / reflect / other).
At the end of the run, print the total and a per-phase breakdown so you can
see which phase burns the most tokens and why.
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


# ---------------------------------------------------------------------------
# Exercise 5: call_llm records usage per phase
# ---------------------------------------------------------------------------
def call_llm(messages, phase="other"):
    """Send messages to Groq. Records token usage into memory['tokens']."""
    response = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=0.2
    )
    usage = getattr(response, "usage", None)
    if usage:
        total = getattr(usage, "total_tokens", 0) or 0
        memory["tokens"]["total"] += total
        memory["tokens"][phase] = memory["tokens"].get(phase, 0) + total
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Knowledge base + RAG
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE = {
    "mars": "Mars is the fourth planet from the Sun. NASA's Perseverance rover landed in Jezero Crater in February 2021.",
    "jupiter": "Jupiter is the largest planet in the solar system, a gas giant famous for the Great Red Spot storm.",
    "saturn": "Saturn is known for its extensive ring system of ice and rock, and has 146 known moons including Titan.",
    "venus": "Venus is the second planet from the Sun and the hottest planet, with surface temperatures around 465°C.",
    "mercury": "Mercury is the smallest planet and the closest to the Sun, completing one orbit every 88 Earth days.",
    "neptune": "Neptune is an ice giant with the fastest winds in the solar system, reaching over 2,000 km/h.",
    "python": "Python is a high-level programming language known for readability. Created by Guido van Rossum in 1991.",
    "groq": "Groq builds fast inference hardware and hosts open-weight LLMs via a cloud API.",
}

REQUIRED_SOURCES = 3


def rag_lookup(query):
    words = set(re.findall(r"[a-z0-9]+", query.lower()))
    hits = []
    for key, text in KNOWLEDGE_BASE.items():
        key_words = set(re.findall(r"[a-z0-9]+", (key + " " + text).lower()))
        if words & key_words:
            hits.append(f"[{key}] {text}")
    return "\n".join(hits) if hits else "no results"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
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
# Memory — includes token counter (Exercise 5)
# ---------------------------------------------------------------------------
memory = {
    "history": [],
    "notes": [],
    "facts": [],
    "sources": [],
    "tokens": {"total": 0},   # <-- Exercise 5: {"total": N, "plan": N, "reflect": N, ...}
}


def reset_run_state():
    memory["history"].clear()
    memory["notes"].clear()
    memory["facts"].clear()
    memory["sources"].clear()
    memory["tokens"] = {"total": 0}


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
# SPAOR — note the `phase=` label on every call_llm call
# ---------------------------------------------------------------------------
def sense(goal, iteration):
    recent_facts = memory["facts"][-5:]
    context = {
        "iteration": iteration,
        "goal": goal,
        "history": memory["history"][-3:],
        "notes": memory["notes"],
        "facts": recent_facts,
        "sources": memory["sources"],
    }
    trace("👁  SENSE", {
        "iteration": iteration,
        "goal": goal,
        "history_len": len(memory["history"]),
        "sources_found": memory["sources"],
        "tokens_so_far": memory["tokens"],
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

Do NOT set COMPLETE until {REQUIRED_SOURCES}+ distinct sources are in your history."""

    user = f"""Goal: {context['goal']}
Iteration: {context['iteration']}
Recent history: {json.dumps(context['history'], default=str)}
Notes: {context['notes']}
Known facts: {context['facts']}
Sources found: {context['sources']} ({len(context['sources'])}/{REQUIRED_SOURCES} needed)

Decide the next action."""

    raw = call_llm(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        phase="plan",   # Exercise 5: label the phase
    )
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
        observation = {"kind": "empty", "ok": False, "summary": "search returned no results"}
    else:
        observation = {"kind": "result", "ok": True, "summary": str(result)}
    trace("🔭 OBSERVE", observation)
    return observation


def reflect(goal, decision, observation):
    if decision.get("action") == "COMPLETE":
        trace("💭 REFLECT", {"done": True, "reason": "agent declared COMPLETE"})
        return True

    prompt = f"""Goal: {goal}
Last action: {decision.get('tool')}({decision.get('args')})
Observation: {observation['summary']}
Observation ok: {observation['ok']}

Rules: if observation ok is false, progress must be false. Never hallucinate.

Reply with JSON only:
{{
  "progress": true or false,
  "fact": "<one factual sentence learned, or empty string>",
  "comment": "<one sentence>"
}}"""

    raw = call_llm(
        [
            {"role": "system", "content": "You judge agent progress. Reply with JSON only. Never hallucinate."},
            {"role": "user", "content": prompt},
        ],
        phase="reflect",   # Exercise 5: label the phase
    )
    judgment = parse_json(raw)
    trace("💭 REFLECT", judgment)

    if judgment.get("progress") and judgment.get("fact"):
        fact = judgment["fact"].strip()
        if fact and fact not in memory["facts"]:
            memory["facts"].append(fact)

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
MAX_ITERATIONS = 10


def print_token_report():
    """Exercise 5: print a breakdown of token usage per phase."""
    tokens = memory["tokens"]
    total = tokens.get("total", 0)
    phases = {k: v for k, v in tokens.items() if k != "total"}
    print("\n📊 TOKEN USAGE REPORT")
    print(f"   Total tokens: {total}")
    for phase, count in sorted(phases.items(), key=lambda x: -x[1]):
        pct = (count / total * 100) if total else 0
        bar = "█" * int(pct / 5)
        print(f"   {phase:12s}: {count:6d}  ({pct:5.1f}%)  {bar}")
    if phases:
        top = max(phases, key=phases.get)
        print(f"\n   → '{top}' burns the most tokens because its prompt includes")
        print(f"     the full tool list, history, and notes — it's the longest context.")


def run_agent(goal):
    reset_run_state()
    print("=" * 60)
    print(f"🎯 GOAL: {goal}")
    print("=" * 60)
    final_answer = None

    for i in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- ITERATION {i} ---")
        context = sense(goal, i)
        decision = plan(context)

        if decision.get("action") == "COMPLETE":
            if len(memory["sources"]) < REQUIRED_SOURCES:
                print(
                    f"\n⛔ COMPLETE rejected: {len(memory['sources'])}/{REQUIRED_SOURCES} sources. Continuing."
                )
                memory["history"].append({
                    "tool": None,
                    "args": None,
                    "observation": f"COMPLETE rejected: need {REQUIRED_SOURCES} sources",
                    "progress": False,
                })
                continue

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
    print(f"Unique sources: {memory['sources']}")
    print_token_report()   # Exercise 5
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
    run_agent("Collect at least 3 distinct facts about planets in the solar system.")

"""
Exercise 4: Enforce a real success condition
---------------------------------------------
Goal: "Collect at least 3 distinct facts about planets in the solar system."
Track unique sources in memory["sources"]. Only allow COMPLETE when 3+ sources
have been found. The agent keeps searching on its own until the bar is met.
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
    "venus": "Venus is the second planet from the Sun and the hottest planet, with surface temperatures around 465°C.",
    "mercury": "Mercury is the smallest planet and the closest to the Sun, completing one orbit every 88 Earth days.",
    "neptune": "Neptune is an ice giant with the fastest winds in the solar system, reaching over 2,000 km/h.",
    "python": "Python is a high-level programming language known for readability. Created by Guido van Rossum in 1991.",
    "groq": "Groq builds fast inference hardware and hosts open-weight LLMs via a cloud API.",
}

REQUIRED_SOURCES = 3   # Exercise 4: the real success bar


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
    # Exercise 4: extract and track unique sources
    if result != "no results":
        for line in result.splitlines():
            m = re.match(r"\[([^\]]+)\]", line)
            if m:
                source = m.group(1)
                if source not in memory["sources"]:
                    memory["sources"].append(source)
                    print(f"  🔖 New source: [{source}]  (total: {len(memory['sources'])})")
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
# Memory — includes `sources` for Exercise 4 and `facts` from Exercise 3
# ---------------------------------------------------------------------------
memory = {
    "history": [],
    "notes": [],
    "facts": [],
    "sources": [],   # <-- unique KB keys that returned real data
}


def reset_run_state():
    memory["history"].clear()
    memory["notes"].clear()
    memory["facts"].clear()
    memory["sources"].clear()


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
    recent_facts = memory["facts"][-5:]
    context = {
        "iteration": iteration,
        "goal": goal,
        "history": memory["history"][-3:],
        "notes": memory["notes"],
        "facts": recent_facts,
        "sources_found": memory["sources"],
        "sources_needed": REQUIRED_SOURCES,
    }
    trace("👁  SENSE", {
        "iteration": iteration,
        "goal": goal,
        "history_len": len(memory["history"]),
        "notes": memory["notes"] or "(empty)",
        "sources_found": memory["sources"],
        f"need {REQUIRED_SOURCES} sources": len(memory["sources"]) >= REQUIRED_SOURCES,
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

IMPORTANT: Do NOT set action to COMPLETE until you have found at least
{REQUIRED_SOURCES} distinct planet sources. Keep searching different planets."""

    user = f"""Goal: {context['goal']}
Iteration: {context['iteration']}
Recent history: {json.dumps(context['history'], default=str)}
Notes: {context['notes']}
Known facts: {context['facts']}
Sources found so far: {context['sources_found']} ({len(context['sources_found'])}/{context['sources_needed']} needed)

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

    raw = call_llm([
        {"role": "system", "content": "You judge agent progress. Reply with JSON only. Never hallucinate."},
        {"role": "user", "content": prompt},
    ])
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
# Loop + helpers — Exercise 4: gate COMPLETE on real success condition
# ---------------------------------------------------------------------------
MAX_ITERATIONS = 10


def run_agent(goal):
    reset_run_state()
    print("=" * 60)
    print(f"🎯 GOAL: {goal}")
    print(f"   Success condition: {REQUIRED_SOURCES}+ distinct planet sources")
    print("=" * 60)
    final_answer = None

    for i in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- ITERATION {i} ---")
        context = sense(goal, i)
        decision = plan(context)

        if decision.get("action") == "COMPLETE":
            # Exercise 4: enforce the real bar before accepting COMPLETE
            if len(memory["sources"]) < REQUIRED_SOURCES:
                print(
                    f"\n⛔ Agent tried to COMPLETE with only "
                    f"{len(memory['sources'])}/{REQUIRED_SOURCES} sources. Forcing continuation."
                )
                memory["history"].append({
                    "tool": None,
                    "args": None,
                    "observation": (
                        f"COMPLETE was rejected: only {len(memory['sources'])} "
                        f"of {REQUIRED_SOURCES} required sources found"
                    ),
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
    print(f"Unique sources: {memory['sources']} ({len(memory['sources'])} total)")
    print(f"Facts learned: {memory['facts']}")
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

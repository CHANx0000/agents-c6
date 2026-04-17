"""
Exercise 6: A second agent as a tool (stretch)
------------------------------------------------
`tool_summarize` is backed by a full SPAOR agent (summarizer_agent.py), not
just a bare LLM call. The main agent calls it like any other tool and receives
a string back — it has no idea a second agent ran inside.

This demonstrates the real "agent-as-tool" pattern:
  - Each agent is a specialist with its own tools, memory, and loop.
  - Agents compose by passing strings in and strings out.
  - The outer agent stays clean and simple.

This file is the complete agent with all prior exercises included.
"""

import json
import os
import re
from datetime import datetime

from dotenv import load_dotenv
from groq import Groq

# Import the specialist agent — this is what makes it "agent-as-tool"
from summarizer_agent import run_summarizer_agent

load_dotenv()

MODEL = "openai/gpt-oss-120b"
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ---------------------------------------------------------------------------
# call_llm with token tracking (Ex 5)
# ---------------------------------------------------------------------------
def call_llm(messages, phase="other"):
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
    "mars": "Mars is the fourth planet from the Sun. NASA's Perseverance rover landed in Jezero Crater in February 2021. Mars has two small moons, Phobos and Deimos. Its surface gravity is about 38% of Earth's.",
    "jupiter": "Jupiter is the largest planet in the solar system, a gas giant famous for the Great Red Spot storm. It has at least 95 known moons, including the four Galilean moons: Io, Europa, Ganymede, and Callisto. Jupiter's mass is more than twice that of all other planets combined.",
    "saturn": "Saturn is known for its extensive ring system of ice and rock, and has 146 known moons including Titan. Titan is the only moon with a dense atmosphere. Saturn's density is so low it would float in water.",
    "venus": "Venus is the second planet from the Sun and the hottest planet, with surface temperatures around 465°C. It rotates backwards compared to most planets. A day on Venus is longer than its year.",
    "mercury": "Mercury is the smallest planet and the closest to the Sun, completing one orbit every 88 Earth days. Despite being closest to the Sun, it is not the hottest planet. Mercury has no atmosphere and no moons.",
    "neptune": "Neptune is an ice giant with the fastest winds in the solar system, reaching over 2,000 km/h. It has 16 known moons, the largest being Triton. Neptune takes 165 Earth years to orbit the Sun.",
    "earth": "Earth is the third planet from the Sun and the only known planet to support life. It has one natural satellite, the Moon. About 71% of Earth's surface is covered by water.",
    "uranus": "Uranus is an ice giant that rotates on its side, with an axial tilt of about 98 degrees. It has 28 known moons and a faint ring system. Uranus was the first planet discovered with a telescope, by William Herschel in 1781.",
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


# Exercise 6: tool_summarize delegates to a full SPAOR agent — agent-as-tool
def tool_summarize(text):
    """
    Summarize text by running the specialist summarizer_agent.

    From the main agent's perspective this is just another tool: put text in,
    get a string out. The fact that a complete SPAOR loop runs inside is
    invisible to the caller — that is the whole point of agent-as-tool.
    """
    if not text or not text.strip():
        return "nothing to summarize"
    return run_summarizer_agent(text)


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
    # Exercise 6: summarize tool backed by a full SPAOR agent
    "summarize": {
        "description": (
            "Summarize a block of text in one sentence using a specialist AI agent. "
            "Pass the full text you want summarized. args: text (string)"
        ),
        "function": tool_summarize,
    },
}

# ---------------------------------------------------------------------------
# Memory (all exercises combined)
# ---------------------------------------------------------------------------
memory = {
    "history": [],
    "notes": [],
    "facts": [],
    "sources": [],
    "tokens": {"total": 0},
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
# SPAOR
# ---------------------------------------------------------------------------
def sense(goal, iteration, enforce_sources=False):
    recent_facts = memory["facts"][-5:]
    context = {
        "iteration": iteration,
        "goal": goal,
        "history": memory["history"][-3:],
        "notes": memory["notes"],
        "facts": recent_facts,
        "sources": memory["sources"],
        "enforce_sources": enforce_sources,   # carried through so plan() can see it
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

    # Only add the sources constraint when the caller actually needs it.
    # Without this guard the prompt tells the model to keep searching for a
    # third source even when the goal has nothing to do with source-counting.
    if context["enforce_sources"]:
        sources_rule = (
            f"\nDo NOT set COMPLETE until you have {REQUIRED_SOURCES}+ distinct "
            f"sources in your history. Keep searching until that bar is met."
        )
        sources_status = (
            f"Sources found: {context['sources']} "
            f"({len(context['sources'])}/{REQUIRED_SOURCES} needed)"
        )
    else:
        sources_rule = (
            "\nIf the goal involves summarizing gathered text, "
            "use the summarize tool before declaring COMPLETE."
        )
        sources_status = f"Sources found so far: {context['sources']}"

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
}}{sources_rule}"""

    user = f"""Goal: {context['goal']}
Iteration: {context['iteration']}
Recent history: {json.dumps(context['history'], default=str)}
Notes: {context['notes']}
Known facts: {context['facts']}
{sources_status}

Decide the next action."""

    raw = call_llm(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        phase="plan",
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
        # Store the FULL result so reflect can reason about everything the tool
        # returned. trace() handles its own display truncation independently.
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
        phase="reflect",
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
    tokens = memory["tokens"]
    total = tokens.get("total", 0)
    phases = {k: v for k, v in tokens.items() if k != "total"}
    print("\n📊 TOKEN USAGE REPORT")
    print(f"   Total tokens: {total}")
    for phase, count in sorted(phases.items(), key=lambda x: -x[1]):
        pct = (count / total * 100) if total else 0
        bar = "█" * int(pct / 5)
        print(f"   {phase:16s}: {count:6d}  ({pct:5.1f}%)  {bar}")


def run_agent(goal, enforce_sources=False):
    reset_run_state()
    print("=" * 60)
    print(f"🎯 GOAL: {goal}")
    print("=" * 60)
    final_answer = None

    for i in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- ITERATION {i} ---")
        context = sense(goal, i, enforce_sources=enforce_sources)
        decision = plan(context)

        if decision.get("action") == "COMPLETE":
            if enforce_sources and len(memory["sources"]) < REQUIRED_SOURCES:
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
    print_token_report()
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
    # Demonstrates the summarize tool (agent-as-tool) in action
    run_agent(
        "Find out about Jupiter and Saturn, then summarize what you learned in one sentence.",
    )

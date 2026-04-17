"""
researcher_agent.py — A specialist agent whose only job is to gather facts.

Same SPAOR shape as summarizer_agent.py. Given a topic, it searches its own
small knowledge base, collects a few distinct facts, and returns them as a
plain string. It has its own isolated memory and tools.

Tools:
  kb_search   — pure Python: look up a topic in the KB
  list_topics — pure Python: show every available topic (helps when stuck)
"""

import json
import os
import re

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = "openai/gpt-oss-120b"
_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# The researcher owns its own small KB so it is a true self-contained
# specialist. A real project would back this with a vector store or an API.
_KB = {
    "mars": "Mars is the fourth planet from the Sun. NASA's Perseverance rover landed in Jezero Crater in February 2021. Mars has two small moons, Phobos and Deimos. Its surface gravity is about 38% of Earth's.",
    "jupiter": "Jupiter is the largest planet in the solar system, a gas giant famous for the Great Red Spot storm. It has at least 95 known moons, including the four Galilean moons: Io, Europa, Ganymede, and Callisto. Jupiter's mass is more than twice that of all other planets combined.",
    "saturn": "Saturn is known for its extensive ring system of ice and rock, and has 146 known moons including Titan. Titan is the only moon with a dense atmosphere. Saturn's density is so low it would float in water.",
    "venus": "Venus is the second planet from the Sun and the hottest planet, with surface temperatures around 465°C. It rotates backwards compared to most planets. A day on Venus is longer than its year.",
    "mercury": "Mercury is the smallest planet and the closest to the Sun, completing one orbit every 88 Earth days. Despite being closest to the Sun, it is not the hottest planet. Mercury has no atmosphere and no moons.",
    "neptune": "Neptune is an ice giant with the fastest winds in the solar system, reaching over 2,000 km/h. It has 16 known moons, the largest being Triton. Neptune takes 165 Earth years to orbit the Sun.",
    "earth": "Earth is the third planet from the Sun and the only known planet to support life. It has one natural satellite, the Moon. About 71% of Earth's surface is covered by water.",
    "uranus": "Uranus is an ice giant that rotates on its side, with an axial tilt of about 98 degrees. It has 28 known moons and a faint ring system. Uranus was the first planet discovered with a telescope, by William Herschel in 1781.",
}

_memory = {
    "history": [],
    "facts": [],
    "tokens": {"total": 0},
}


def _call_llm(messages, phase="other"):
    response = _client.chat.completions.create(
        model=MODEL, messages=messages, temperature=0.2
    )
    usage = getattr(response, "usage", None)
    if usage:
        total = getattr(usage, "total_tokens", 0) or 0
        _memory["tokens"]["total"] += total
        _memory["tokens"][phase] = _memory["tokens"].get(phase, 0) + total
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Tools — pure Python, no LLM calls
# ---------------------------------------------------------------------------
def _tool_kb_search(topic):
    if not topic:
        return "no results"
    words = set(re.findall(r"[a-z0-9]+", topic.lower()))
    hits = []
    for key, text in _KB.items():
        key_words = set(re.findall(r"[a-z0-9]+", (key + " " + text).lower()))
        if words & key_words:
            hits.append(f"[{key}] {text}")
    return "\n".join(hits) if hits else "no results"


def _tool_list_topics(_ignored=None):
    return "Available topics: " + ", ".join(sorted(_KB.keys()))


_TOOLS = {
    "kb_search": {
        "description": "Look up a topic in the knowledge base. args: topic (string)",
        "function": _tool_kb_search,
    },
    "list_topics": {
        "description": "List every topic available in the KB. Pass an empty string.",
        "function": _tool_list_topics,
    },
}


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------
def _trace(phase, data):
    print(f"\n  [RESEARCHER] {phase}")
    for key, value in data.items():
        preview = str(value)
        if len(preview) > 100:
            preview = preview[:97] + "..."
        print(f"    {key}: {preview}")


# ---------------------------------------------------------------------------
# SPAOR
# ---------------------------------------------------------------------------
MIN_FACTS = 2


def _sense(goal, iteration):
    context = {
        "iteration": iteration,
        "goal": goal,
        "history": _memory["history"][-3:],
        "facts": _memory["facts"],
    }
    _trace("👁  SENSE", {
        "iteration": iteration,
        "goal": goal,
        "facts_collected": len(_memory["facts"]),
    })
    return context


def _plan(context):
    tool_list = "\n".join(
        f"  - {name}: {info['description']}" for name, info in _TOOLS.items()
    )
    system = f"""You are a research specialist. Your only job is to collect
distinct factual snippets about the given topic from your KB.

Available tools:
{tool_list}

Recommended workflow:
1. Use kb_search one or more times to pull text about the topic.
2. When you have at least {MIN_FACTS} distinct facts, declare COMPLETE and put
   them all in the answer field, joined by newlines.

CRITICAL RULES:
- You MUST use kb_search at least once BEFORE declaring COMPLETE.
- Do NOT declare COMPLETE if "Facts collected so far" below is empty ([]).
  You have zero facts until kb_search returns them.
- Do NOT invent or assume facts. Only use what kb_search returns.

Reply with ONE JSON object only:
{{
  "action": "USE_TOOL" or "COMPLETE",
  "tool": "<tool name or null>",
  "args": "<single string argument or null>",
  "reasoning": "<why>",
  "answer": "<collected facts if COMPLETE, else null>"
}}"""

    user = f"""Topic to research: {context['goal']}
Iteration: {context['iteration']}
History: {json.dumps(context['history'], default=str)}
Facts collected so far: {context['facts']}

Decide the next action."""

    raw = _call_llm(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        phase="plan",
    )
    decision = _parse_json(raw)
    _trace("🧠 PLAN", {
        "action": decision.get("action"),
        "tool": decision.get("tool"),
        "reasoning": decision.get("reasoning"),
    })
    return decision


def _act(decision):
    if decision.get("action") != "USE_TOOL":
        return None
    name = decision.get("tool")
    if name not in _TOOLS:
        result = f"ERROR: unknown tool '{name}'"
    else:
        try:
            result = _TOOLS[name]["function"](decision.get("args"))
        except Exception as e:
            result = f"ERROR: {e}"
    _trace("⚡ ACT", {"tool": name, "result": result})
    return result


def _observe(decision, result):
    if decision.get("action") != "USE_TOOL":
        obs = {"kind": "none", "ok": True, "summary": "no action taken"}
    elif isinstance(result, str) and result.startswith("ERROR"):
        obs = {"kind": "error", "ok": False, "summary": result}
    elif isinstance(result, str) and result.strip() == "no results":
        obs = {"kind": "empty", "ok": False, "summary": "kb_search returned no results"}
    else:
        obs = {"kind": "result", "ok": True, "summary": str(result)}
    _trace("🔭 OBSERVE", obs)
    return obs


def _reflect(decision, observation):
    if decision.get("action") == "COMPLETE":
        _trace("💭 REFLECT", {"done": True})
        return True

    # Harvest every [source] line as a distinct fact — deterministic, no LLM.
    if observation["ok"] and observation["kind"] == "result":
        for line in observation["summary"].splitlines():
            if line.startswith("[") and line not in _memory["facts"]:
                _memory["facts"].append(line)

    _memory["history"].append({
        "tool": decision.get("tool"),
        "observation": observation["summary"][:200],
        "progress": observation["ok"],
    })
    return False


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------
MAX_ITERATIONS = 6


def run_researcher_agent(topic):
    """Run the researcher and return collected facts as a newline-joined string."""
    _memory["history"].clear()
    _memory["facts"].clear()

    print("\n  " + "=" * 56)
    print(f"  [RESEARCHER AGENT STARTED] topic={topic}")
    print("  " + "=" * 56)

    answer = None
    for i in range(1, MAX_ITERATIONS + 1):
        print(f"\n  [RESEARCHER] --- iteration {i} ---")
        context = _sense(topic, i)
        decision = _plan(context)

        if decision.get("action") == "COMPLETE":
            if len(_memory["facts"]) < MIN_FACTS:
                print(
                    f"\n  [RESEARCHER] ⛔ COMPLETE rejected: "
                    f"{len(_memory['facts'])}/{MIN_FACTS} facts collected. "
                    f"Must search first."
                )
                _memory["history"].append({
                    "tool": None,
                    "observation": f"COMPLETE rejected: need {MIN_FACTS} facts, have {len(_memory['facts'])}",
                    "progress": False,
                })
                continue
            answer = decision.get("answer") or "\n".join(_memory["facts"])
            _reflect(decision, {"kind": "none", "ok": True, "summary": ""})
            break

        result = _act(decision)
        observation = _observe(decision, result)
        if _reflect(decision, observation):
            break
    else:
        answer = "\n".join(_memory["facts"]) or "(researcher hit max iterations)"

    # Fallback: if the LLM forgot to fill `answer`, use harvested facts.
    if not answer or not answer.strip():
        answer = "\n".join(_memory["facts"]) or "(no facts found)"

    print(f"\n  [RESEARCHER AGENT DONE] → {len(_memory['facts'])} facts\n")
    return answer


def _parse_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"action": "COMPLETE", "answer": text.strip()}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"action": "COMPLETE", "answer": text.strip()}


if __name__ == "__main__":
    out = run_researcher_agent("jupiter and saturn")
    print("\nResearcher output:\n" + out)

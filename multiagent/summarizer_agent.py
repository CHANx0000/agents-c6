"""
summarizer_agent.py — A specialist agent whose only job is to summarize text.

This is a self-contained mini-agent built from the same SPAOR pattern as
simple_agent.py. It has its own tools, its own memory, and its own loop.
It is designed to be called by another agent via run_summarizer_agent(text).

Tools this agent has:
  split_text   — pure Python: break text into numbered sentences
  find_topic   — pure Python: find the most common meaningful word
  write_summary — one LLM call: produce a one-sentence summary

The agent works through these steps, then declares COMPLETE with its summary.
"""

import os
import re
from collections import Counter

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = "openai/gpt-oss-120b"
_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------------------------------------------------------------------
# The summarizer agent has its own isolated memory and token counter.
# This keeps it independent of the main agent — no shared state.
# ---------------------------------------------------------------------------
_memory = {
    "history": [],
    "tokens": {"total": 0},
}

# ---------------------------------------------------------------------------
# LLM call — scoped to this module so token costs are tracked separately
# ---------------------------------------------------------------------------
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
# Summarizer-specific tools (no LLM needed for the first two)
# ---------------------------------------------------------------------------
def _tool_split_text(text):
    """Split the input into numbered sentences. Pure Python — no LLM."""
    # Split on sentence-ending punctuation followed by a space or end of string
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in raw if s.strip()]
    if not sentences:
        return "no sentences found"
    lines = [f"{i + 1}. {s}" for i, s in enumerate(sentences)]
    return f"{len(sentences)} sentence(s) found:\n" + "\n".join(lines)


def _tool_find_topic(text):
    """Find the most frequently mentioned meaningful word. Pure Python."""
    # Common English stop words to ignore
    stop_words = {
        "the", "a", "an", "is", "it", "in", "on", "of", "for", "and", "or",
        "to", "with", "its", "has", "are", "was", "by", "be", "from", "as",
        "at", "this", "that", "which", "have", "had", "not", "but", "were",
    }
    words = re.findall(r"[a-z]+", text.lower())
    meaningful = [w for w in words if w not in stop_words and len(w) > 3]
    if not meaningful:
        return "no clear topic found"
    most_common = Counter(meaningful).most_common(3)
    topics = ", ".join(f'"{w}" ({c}x)' for w, c in most_common)
    return f"Top topic words: {topics}"


def _tool_write_summary(text):
    """Ask the LLM to write a one-sentence summary. This is the only LLM call."""
    if not text or not text.strip():
        return "nothing to summarize"
    reply = _call_llm(
        [
            {
                "role": "system",
                "content": (
                    "You are a summarization specialist. "
                    "Write exactly ONE clear, informative sentence that captures "
                    "the most important idea in the user's text. "
                    "No preamble, no filler — just the sentence."
                ),
            },
            {"role": "user", "content": text},
        ],
        phase="write_summary",
    )
    return reply.strip()


_TOOLS = {
    "split_text": {
        "description": "Split the text into numbered sentences. args: text (string)",
        "function": _tool_split_text,
    },
    "find_topic": {
        "description": "Find the most frequently mentioned meaningful words in the text. args: text (string)",
        "function": _tool_find_topic,
    },
    "write_summary": {
        "description": "Write a one-sentence summary of the text using AI. args: text (string)",
        "function": _tool_write_summary,
    },
}


# ---------------------------------------------------------------------------
# Tracing — prefixed with [SUMMARIZER] so you can tell the two agents apart
# ---------------------------------------------------------------------------
def _trace(phase, data):
    print(f"\n  [SUMMARIZER] {phase}")
    for key, value in data.items():
        preview = str(value)
        if len(preview) > 100:
            preview = preview[:97] + "..."
        print(f"    {key}: {preview}")


# ---------------------------------------------------------------------------
# SPAOR — same pattern as simple_agent.py, just scoped to summarization
# ---------------------------------------------------------------------------
import json

def _sense(goal, iteration):
    context = {
        "iteration": iteration,
        "goal": goal,
        "history": _memory["history"][-3:],
    }
    _trace("👁  SENSE", {
        "iteration": iteration,
        "goal": goal,
        "history_len": len(_memory["history"]),
    })
    return context


def _plan(context):
    tool_list = "\n".join(
        f"  - {name}: {info['description']}" for name, info in _TOOLS.items()
    )
    system = f"""You are a text summarization agent. Your only job is to produce
    a one-sentence summary of the text you receive.

    Available tools:
    {tool_list}

    Recommended workflow:
    1. Use split_text to understand the structure of the input.
    2. Use find_topic to identify the main subject.
    3. Use write_summary to produce the final sentence.
    4. Declare COMPLETE with the summary as the answer.

    On every turn reply with ONE JSON object and nothing else:
    {{
    "action": "USE_TOOL" or "COMPLETE",
    "tool": "<tool name or null>",
    "args": "<single string argument or null>",
    "reasoning": "<why you chose this>",
    "answer": "<final one-sentence summary if COMPLETE, else null>"
    }}"""

    user = f"""Text to summarize: {context['goal']}
    Iteration: {context['iteration']}
    History: {json.dumps(context['history'], default=str)}

    What is your next step?"""

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
        _trace("⚡ ACT", {"result": "no tool — chose COMPLETE"})
        return None
    tool_name = decision.get("tool")
    if tool_name not in _TOOLS:
        result = f"ERROR: unknown tool '{tool_name}'"
        _trace("⚡ ACT", {"result": result})
        return result
    try:
        result = _TOOLS[tool_name]["function"](decision.get("args"))
    except Exception as e:
        result = f"ERROR: {e}"
    _trace("⚡ ACT", {"tool": tool_name, "result": result})
    return result


def _observe(decision, result):
    if decision.get("action") != "USE_TOOL":
        obs = {"kind": "none", "ok": True, "summary": "no action taken"}
    elif isinstance(result, str) and result.startswith("ERROR"):
        obs = {"kind": "error", "ok": False, "summary": result}
    else:
        obs = {"kind": "result", "ok": True, "summary": str(result)}
    _trace("🔭 OBSERVE", obs)
    return obs


def _reflect(goal, decision, observation):
    if decision.get("action") == "COMPLETE":
        _trace("💭 REFLECT", {"done": True})
        return True

    prompt = f"""Summarization goal: {goal}
    Last action: {decision.get('tool')}
    Observation: {observation['summary']}
    Observation ok: {observation['ok']}

    Did this step bring us closer to writing a one-sentence summary?
    Reply with JSON only: {{"progress": true or false, "comment": "<one sentence>"}}"""

    raw = _call_llm(
        [
            {"role": "system", "content": "You judge summarization progress. Reply JSON only."},
            {"role": "user", "content": prompt},
        ],
        phase="reflect",
    )
    judgment = _parse_json(raw)
    _trace("💭 REFLECT", judgment)

    _memory["history"].append({
        "tool": decision.get("tool"),
        "observation": observation["summary"],
        "progress": judgment.get("progress"),
    })
    return False


# ---------------------------------------------------------------------------
# Public interface — this is what the main agent calls
# ---------------------------------------------------------------------------
MAX_ITERATIONS = 5


def run_summarizer_agent(text):
    """
    Run the summarizer agent on `text` and return a one-sentence summary.

    This is intentionally the same signature as any other tool function:
    takes a string, returns a string. The caller does not need to know
    that a full SPAOR loop is running inside.
    """
    # Reset per-run state (keep token totals for reporting)
    _memory["history"].clear()

    print("\n" + "  " + "=" * 56)
    print(f"  [SUMMARIZER AGENT STARTED]")
    print("  " + "=" * 56)

    summary = None

    for i in range(1, MAX_ITERATIONS + 1):
        print(f"\n  [SUMMARIZER] --- iteration {i} ---")
        context = _sense(text, i)
        decision = _plan(context)

        if decision.get("action") == "COMPLETE":
            summary = decision.get("answer") or "(no summary produced)"
            _reflect(text, decision, {"kind": "none", "ok": True, "summary": ""})
            break

        result = _act(decision)
        observation = _observe(decision, result)
        done = _reflect(text, decision, observation)
        if done:
            break
    else:
        summary = "(summarizer hit max iterations)"

    print(f"\n  [SUMMARIZER AGENT DONE] → {summary}")
    print("  " + "=" * 56 + "\n")
    return summary or "(no summary produced)"


def _parse_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"action": "COMPLETE", "answer": text.strip()}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"action": "COMPLETE", "answer": text.strip()}


# ---------------------------------------------------------------------------
# Run standalone for testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = (
        "Jupiter is the largest planet in the solar system, a gas giant famous "
        "for the Great Red Spot storm. Saturn is known for its extensive ring "
        "system of ice and rock, and has 146 known moons including Titan."
    )
    result = run_summarizer_agent(sample)
    print(f"\nFinal summary: {result}")
    print(f"Summarizer token usage: {_memory['tokens']}")

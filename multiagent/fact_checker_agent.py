"""
fact_checker_agent.py — A specialist agent whose only job is to verify claims.

Same SPAOR shape as the other specialists. Given a draft text, it extracts
claims, checks each one against a knowledge base, and returns a verdict:
either "VERIFIED: <text>" or "WEAK: <claim> — <reason>".

Tools:
  extract_claims — pure Python: pull out factual assertions from text
  check_claim    — pure Python: look up a claim against the KB
  verdict        — LLM call: produce a final verified/weak judgment
"""

import json
import os
import re

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = "openai/gpt-oss-120b"
_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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
    "claims": [],
    "checked": [],
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
# Tools
# ---------------------------------------------------------------------------
def _tool_extract_claims(text):
    """Split text into individual factual claims. Pure Python heuristic."""
    if not text or not text.strip():
        return "no text provided"
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    claims = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    if not claims:
        return "no claims found"
    _memory["claims"] = claims
    lines = [f"{i + 1}. {c}" for i, c in enumerate(claims)]
    return f"{len(claims)} claim(s) extracted:\n" + "\n".join(lines)


_STOP_WORDS = {
    "the", "a", "an", "is", "it", "in", "on", "of", "for", "and", "or",
    "to", "with", "its", "has", "are", "was", "by", "be", "from", "as",
    "at", "this", "that", "which", "have", "had", "not", "but", "were",
    "planet", "known", "system", "solar", "most", "than", "also", "about",
}


def _tool_check_claim(claim):
    """Check a single claim against the KB. Returns supporting evidence or 'no evidence'."""
    if not claim:
        return "no evidence"
    claim_words = set(re.findall(r"[a-z0-9]+", claim.lower())) - _STOP_WORDS
    evidence = []
    for key, text in _KB.items():
        # The topic key itself must appear in the claim for a match
        if key not in claim_words:
            continue
        kb_words = set(re.findall(r"[a-z0-9]+", text.lower())) - _STOP_WORDS
        overlap = claim_words & kb_words
        if len(overlap) >= 2:
            evidence.append(f"[{key}] {text}")
    if evidence:
        _memory["checked"].append({"claim": claim, "supported": True, "evidence": evidence})
        return "SUPPORTED — " + " | ".join(evidence)
    _memory["checked"].append({"claim": claim, "supported": False, "evidence": []})
    return "no evidence found in KB"


def _tool_verdict(text):
    """Ask the LLM to produce a final fact-check verdict. Single LLM call."""
    if not text or not text.strip():
        return "nothing to verify"
    checked_summary = json.dumps(_memory["checked"], indent=2) if _memory["checked"] else "none"
    reply = _call_llm(
        [
            {
                "role": "system",
                "content": (
                    "You are a fact-check specialist. Based on the claim checks "
                    "provided, produce a short verdict. For each claim say VERIFIED "
                    "or WEAK. End with an overall status: ALL VERIFIED or HAS WEAK CLAIMS. "
                    "Be concise — no preamble."
                ),
            },
            {
                "role": "user",
                "content": f"Original text: {text}\n\nClaim checks:\n{checked_summary}",
            },
        ],
        phase="verdict",
    )
    return reply.strip()


_TOOLS = {
    "extract_claims": {
        "description": "Extract factual claims from the text. args: text (string)",
        "function": _tool_extract_claims,
    },
    "check_claim": {
        "description": "Check a single claim against the knowledge base. args: claim (string)",
        "function": _tool_check_claim,
    },
    "verdict": {
        "description": "Produce a final fact-check verdict using AI. args: original text (string)",
        "function": _tool_verdict,
    },
}


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------
def _trace(phase, data):
    print(f"\n  [FACT-CHECKER] {phase}")
    for key, value in data.items():
        preview = str(value)
        if len(preview) > 100:
            preview = preview[:97] + "..."
        print(f"    {key}: {preview}")


# ---------------------------------------------------------------------------
# SPAOR
# ---------------------------------------------------------------------------
def _sense(goal, iteration):
    context = {
        "iteration": iteration,
        "goal": goal,
        "history": _memory["history"][-3:],
        "claims": _memory["claims"],
        "checked": _memory["checked"],
    }
    _trace("SENSE", {
        "iteration": iteration,
        "goal": goal,
        "claims_extracted": len(_memory["claims"]),
        "claims_checked": len(_memory["checked"]),
    })
    return context


def _plan(context):
    tool_list = "\n".join(
        f"  - {name}: {info['description']}" for name, info in _TOOLS.items()
    )
    system = f"""You are a fact-checking agent. Your only job is to verify the
factual claims in the text you receive.

Available tools:
{tool_list}

Recommended workflow:
1. Use extract_claims to identify individual claims.
2. Use check_claim for EACH claim (one call per claim).
3. Use verdict to produce the final judgment.
4. Declare COMPLETE with the verdict as the answer.

Reply with ONE JSON object only:
{{
  "action": "USE_TOOL" or "COMPLETE",
  "tool": "<tool name or null>",
  "args": "<single string argument or null>",
  "reasoning": "<why>",
  "answer": "<final verdict if COMPLETE, else null>"
}}"""

    user = f"""Text to fact-check: {context['goal']}
Iteration: {context['iteration']}
History: {json.dumps(context['history'], default=str)}
Claims extracted: {context['claims']}
Claims checked so far: {json.dumps(context['checked'], default=str)}

What is your next step?"""

    raw = _call_llm(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        phase="plan",
    )
    decision = _parse_json(raw)
    _trace("PLAN", {
        "action": decision.get("action"),
        "tool": decision.get("tool"),
        "reasoning": decision.get("reasoning"),
    })
    return decision


def _act(decision):
    if decision.get("action") != "USE_TOOL":
        _trace("ACT", {"result": "no tool — chose COMPLETE"})
        return None
    tool_name = decision.get("tool")
    if tool_name not in _TOOLS:
        result = f"ERROR: unknown tool '{tool_name}'"
        _trace("ACT", {"result": result})
        return result
    try:
        result = _TOOLS[tool_name]["function"](decision.get("args"))
    except Exception as e:
        result = f"ERROR: {e}"
    _trace("ACT", {"tool": tool_name, "result": result})
    return result


def _observe(decision, result):
    if decision.get("action") != "USE_TOOL":
        obs = {"kind": "none", "ok": True, "summary": "no action taken"}
    elif isinstance(result, str) and result.startswith("ERROR"):
        obs = {"kind": "error", "ok": False, "summary": result}
    else:
        obs = {"kind": "result", "ok": True, "summary": str(result)}
    _trace("OBSERVE", obs)
    return obs


def _reflect(decision, observation):
    if decision.get("action") == "COMPLETE":
        _trace("REFLECT", {"done": True})
        return True

    _memory["history"].append({
        "tool": decision.get("tool"),
        "observation": observation["summary"][:200],
        "progress": observation["ok"],
    })
    return False


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------
MAX_ITERATIONS = 8


def run_fact_checker_agent(text):
    """Run the fact checker on `text` and return a verdict string."""
    _memory["history"].clear()
    _memory["claims"].clear()
    _memory["checked"].clear()

    print("\n  " + "=" * 56)
    print(f"  [FACT-CHECKER AGENT STARTED]")
    print("  " + "=" * 56)

    verdict = None

    for i in range(1, MAX_ITERATIONS + 1):
        print(f"\n  [FACT-CHECKER] --- iteration {i} ---")
        context = _sense(text, i)
        decision = _plan(context)

        if decision.get("action") == "COMPLETE":
            verdict = decision.get("answer") or "(no verdict produced)"
            _reflect(decision, {"kind": "none", "ok": True, "summary": ""})
            break

        result = _act(decision)
        observation = _observe(decision, result)
        if _reflect(decision, observation):
            break
    else:
        verdict = "(fact-checker hit max iterations)"

    print(f"\n  [FACT-CHECKER AGENT DONE] → {verdict}")
    print("  " + "=" * 56 + "\n")
    return verdict or "(no verdict produced)"


def _parse_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"action": "COMPLETE", "answer": text.strip()}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"action": "COMPLETE", "answer": text.strip()}


if __name__ == "__main__":
    sample = (
        "Jupiter is the largest planet in the solar system. "
        "Saturn has 146 known moons including Titan. "
        "Mars has a population of 3 million people."
    )
    result = run_fact_checker_agent(sample)
    print(f"\nFinal verdict: {result}")
    print(f"Fact-checker token usage: {_memory['tokens']}")

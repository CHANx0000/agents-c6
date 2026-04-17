"""
Exercise 8: The Handoff Pattern
--------------------------------
In the manager pattern the outer agent CALLS a specialist and waits for a
string back — like a function call. The caller stays in charge.

In the handoff pattern the active agent TRANSFERS control to another agent
and steps out of the loop entirely. The new agent owns the rest of the task.
It may finish the job itself, or hand off again to yet another specialist.
Control never comes back.

                  ┌──────────┐
    user goal ──▶ │  TRIAGE  │  (decides "who should own this?")
                  └────┬─────┘
                       │  HANDOFF
                       ▼
                 ┌─────────────┐
                 │  RESEARCHER │  (runs full SPAOR, produces facts)
                 └──────┬──────┘
                        │  HANDOFF
                        ▼
                 ┌──────────────┐
                 │ FACT-CHECKER │  (verifies claims against KB)
                 └──────┬───────┘
                        │  HANDOFF (can go back to researcher if weak)
                        ▼
                 ┌──────────────┐
                 │  SUMMARIZER  │  (produces final one-sentence answer)
                 └──────┬───────┘
                        │
                        ▼
                  final answer to user

Key differences from ex7_manager_pattern.py:
  - Triage has NO loop and NO tools. It is a single LLM call that outputs a
    routing decision. That is the whole agent.
  - After triage, control is transferred. Triage is not invoked again.
  - A specialist can itself trigger another handoff (researcher → fact_checker
    → summarizer) without consulting triage.
  - The fact-checker can hand off BACK to the researcher if evidence is weak,
    creating a tight loop without involving triage.

The orchestrator below is a while loop that just keeps pulling the "next
agent" off a queue until someone returns a `done` result.
"""

import json
import os
import re

from dotenv import load_dotenv
from groq import Groq

from researcher_agent import run_researcher_agent
from summarizer_agent import run_summarizer_agent
from fact_checker_agent import run_fact_checker_agent

load_dotenv()

MODEL = "openai/gpt-oss-120b"
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def call_llm(messages):
    response = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=0.2
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Triage agent — NO SPAOR loop, NO tools. One LLM call, one routing decision.
# That minimal shape is the point: a triage agent is just a classifier with
# a handoff target attached to each class.
# ---------------------------------------------------------------------------
TRIAGE_TARGETS = ["researcher", "summarizer", "fact_checker"]


def triage(user_goal):
    system = f"""You are a TRIAGE agent. You do not do any work yourself —
you choose which specialist should own the task and then hand off control.

Specialists you can hand off to:
  - researcher: given a TOPIC, pulls facts from a knowledge base.
  - fact_checker: given TEXT with claims, verifies each claim against a KB.
  - summarizer: given a TEXT, compresses it into one sentence.

Reply with ONE JSON object only:
{{
  "target": "<researcher or fact_checker or summarizer>",
  "handoff_goal": "<the goal restated in the form that specialist expects>",
  "reasoning": "<one sentence>"
}}

Rules:
- If the user wants research (even if they also want it summarized later),
  hand off to researcher first. The researcher can hand off to the
  fact-checker and then the summarizer itself when it finishes.
- If the user already has text and wants it verified, hand off to fact_checker.
- If the user already has verified text and just wants compression, hand off
  straight to summarizer.
"""
    raw = call_llm(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": f"User goal: {user_goal}"},
        ]
    )
    decision = _parse_json(raw)
    print(f"\n[TRIAGE] decision: {decision}")
    return decision


# ---------------------------------------------------------------------------
# Specialist wrappers — each returns the same contract:
#   {"kind": "done", "answer": str}              (terminal)
#   {"kind": "handoff", "target": str, "goal": str}   (pass control on)
#
# Wrappers exist so the underlying agents (which just return strings) can
# participate in the handoff protocol without being modified.
# ---------------------------------------------------------------------------
def run_researcher_with_handoff(goal, user_goal):
    facts = run_researcher_agent(goal)

    # Decide whether to hand off further. If the user goal mentions
    # verification or summarization, chain to fact_checker first.
    if _wants_verification(user_goal) or _wants_summary(user_goal):
        print("\n[HANDOFF] researcher → fact_checker")
        return {"kind": "handoff", "target": "fact_checker", "goal": facts}
    return {"kind": "done", "answer": facts}


def run_fact_checker_with_handoff(goal, user_goal):
    verdict = run_fact_checker_agent(goal)

    # If the verdict mentions weak claims, hand back to researcher for
    # more evidence — this is the tight loop the lecture describes.
    if "WEAK" in verdict.upper() and not _already_retried(user_goal):
        print("\n[HANDOFF] fact_checker → researcher (weak claims found)")
        return {"kind": "handoff", "target": "researcher", "goal": goal}

    # Otherwise chain to summarizer if the user wants a summary.
    if _wants_summary(user_goal):
        print("\n[HANDOFF] fact_checker → summarizer")
        combined = f"{goal}\n\nFact-check verdict: {verdict}"
        return {"kind": "handoff", "target": "summarizer", "goal": combined}
    return {"kind": "done", "answer": verdict}


def run_summarizer_with_handoff(goal, user_goal):
    summary = run_summarizer_agent(goal)
    return {"kind": "done", "answer": summary}


def _wants_summary(user_goal):
    text = (user_goal or "").lower()
    return any(word in text for word in ("summar", "one sentence", "tl;dr", "brief"))


def _wants_verification(user_goal):
    text = (user_goal or "").lower()
    return any(word in text for word in ("verif", "fact", "check", "accurate"))


_retried = False

def _already_retried(user_goal):
    """Prevent infinite fact_checker ↔ researcher loops (max one retry)."""
    global _retried
    if _retried:
        return True
    _retried = True
    return False


AGENTS = {
    "researcher": run_researcher_with_handoff,
    "fact_checker": run_fact_checker_with_handoff,
    "summarizer": run_summarizer_with_handoff,
}


# ---------------------------------------------------------------------------
# Orchestrator — the only thing that understands handoffs. Every participating
# agent just returns {"kind": "done"} or {"kind": "handoff"}.
# ---------------------------------------------------------------------------
MAX_HANDOFFS = 6


def run_with_handoff(user_goal):
    global _retried
    _retried = False

    print("=" * 60)
    print(f"🎯 USER GOAL: {user_goal}")
    print("=" * 60)

    decision = triage(user_goal)
    current_target = decision.get("target")
    current_goal = decision.get("handoff_goal") or user_goal

    if current_target not in AGENTS:
        print(f"⛔ Triage picked invalid target '{current_target}'. Aborting.")
        return None

    print(f"\n[HANDOFF] triage → {current_target}")

    for hop in range(1, MAX_HANDOFFS + 1):
        agent_fn = AGENTS[current_target]
        result = agent_fn(current_goal, user_goal)

        if result["kind"] == "done":
            print("\n" + "=" * 60)
            print(f"FINAL ANSWER (from {current_target}): {result['answer']}")
            print(f"Handoffs used: {hop}")
            print("=" * 60)
            return result["answer"]

        if result["kind"] == "handoff":
            current_target = result["target"]
            current_goal = result["goal"]
            if current_target not in AGENTS:
                print(f"⛔ Invalid handoff target '{current_target}'. Aborting.")
                return None
            continue

        print(f"⛔ Unknown result kind: {result}")
        return None

    print(f"⚠️  Hit max handoffs ({MAX_HANDOFFS}) without a terminal result")
    return None


def _parse_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


if __name__ == "__main__":
    # Expected path: triage → researcher → fact_checker → summarizer → done
    run_with_handoff(
        "Research Jupiter and Saturn, verify the facts, and give me a one-sentence summary."
    )

    # Expected path: triage → fact_checker → summarizer → done
    run_with_handoff(
        "Verify and summarize this: Mars is the fourth planet from the Sun. "
        "NASA's Perseverance rover landed in Jezero Crater in 2021."
    )

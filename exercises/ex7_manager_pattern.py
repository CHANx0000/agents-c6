"""
Exercise 7: The Manager Pattern
--------------------------------
The manager has NO direct tools of its own. Its entire toolbox is other
agents. Its job is to decide *which specialist to call, in what order, with
what input*, and to combine their outputs into a final answer.

Compare with ex6_agent_as_tool.py:
  - ex6: main agent has normal tools (search, calculate, ...) AND one
         agent-tool (summarize). The specialist is an extra helper.
  - ex7: the manager has ONLY agent-tools. It is pure orchestration.

Pattern recap:

        +-----------+
        |  MANAGER  |     plans the workflow
        +-----+-----+
              |
     +--------+--------+
     |                 |
  researcher       summarizer     <- full SPAOR agents, each a specialist
     |                 |
   (kb_search)    (split_text,
   (list_topics)   find_topic,
                   write_summary)

The manager never touches a KB or a calculator directly. It delegates.
"""

import json
import os
import re

from dotenv import load_dotenv
from groq import Groq

# Both tools ARE full agents — the manager has nothing else.
from researcher_agent import run_researcher_agent
from summarizer_agent import run_summarizer_agent

load_dotenv()

MODEL = "openai/gpt-oss-120b"
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


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
# Manager's "tools" are specialist agents. Each wrapper is one line — it
# simply forwards the argument string to the specialist's public entry point.
# ---------------------------------------------------------------------------
def tool_researcher(topic):
    if not topic or not topic.strip():
        return "ERROR: researcher needs a non-empty topic"
    return run_researcher_agent(topic)


def tool_summarizer(text):
    if not text or not text.strip():
        return "ERROR: summarizer needs non-empty text"
    return run_summarizer_agent(text)


TOOLS = {
    "researcher": {
        "description": (
            "Delegate to the research specialist. It searches its KB and "
            "returns a block of collected facts. args: topic (string)"
        ),
        "function": tool_researcher,
    },
    "summarizer": {
        "description": (
            "Delegate to the summarization specialist. It compresses a block "
            "of text into one clear sentence. args: text (string)"
        ),
        "function": tool_summarizer,
    },
}


memory = {
    "history": [],
    "last_research": "",   # full researcher output (passed to summarizer via
                           # the LAST_RESEARCH sentinel — keeps manager prompt small)
    "last_summary": "",    # full summarizer output (once set, the manager
                           # MUST COMPLETE with this text as its answer)
    "tokens": {"total": 0},
}


def reset_run_state():
    memory["history"].clear()
    memory["last_research"] = ""
    memory["last_summary"] = ""
    memory["tokens"] = {"total": 0}


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------
def trace(phase, data):
    print(f"\n{phase}")
    for key, value in data.items():
        preview = str(value)
        if len(preview) > 140:
            preview = preview[:137] + "..."
        print(f"  {key}: {preview}")


# ---------------------------------------------------------------------------
# SPAOR — the manager's loop. Identical shape to every other SPAOR agent,
# but the "tools" happen to be full agents underneath.
# ---------------------------------------------------------------------------
def sense(goal, iteration):
    context = {
        "iteration": iteration,
        "goal": goal,
        "history": memory["history"][-3:],
        "has_research": bool(memory["last_research"]),
        "has_summary": bool(memory["last_summary"]),
        "last_summary": memory["last_summary"],
    }
    trace("👁  SENSE (manager)", {
        "iteration": iteration,
        "goal": goal,
        "history_len": len(memory["history"]),
        "has_research": context["has_research"],
        "has_summary": context["has_summary"],
    })
    return context


def plan(context):
    tool_list = "\n".join(
        f"- {name}: {info['description']}" for name, info in TOOLS.items()
    )
    system = f"""You are a MANAGER agent. You do not execute work yourself —
you delegate to specialist agents. Your only tools are specialist agents.

Available specialists:
{tool_list}

Typical workflow for a "research and summarize" goal:
  1. Call researcher with a topic.
  2. Call summarizer with the researcher's output (passed verbatim).
  3. Declare COMPLETE with the summary as the answer.

Rules you MUST follow:
- Never invent facts. If you need information, call researcher.
- When you call summarizer, the "args" field should be the word "LAST_RESEARCH"
  and the orchestrator will substitute the most recent researcher output.
  This keeps your JSON short and avoids copying long text into your own prompt.
- If a summary is already available (shown below as "FINAL SUMMARY"), you are
  DONE. Set action to COMPLETE and copy the FINAL SUMMARY verbatim into the
  "answer" field. Do NOT call any more specialists.
- Do NOT call the researcher a second time — one call is enough. The
  researcher's output is never "truncated"; history previews may look short
  but the full text is already stored and will be passed to the summarizer.

Reply with ONE JSON object only:
{{
  "action": "USE_TOOL" or "COMPLETE",
  "tool": "<specialist name or null>",
  "args": "<string argument or 'LAST_RESEARCH' or null>",
  "reasoning": "<why>",
  "answer": "<final answer if COMPLETE, else null>"
}}"""

    summary_block = (
        f"FINAL SUMMARY (already produced — COMPLETE with this verbatim):\n"
        f"{context['last_summary']}\n"
        if context["has_summary"]
        else "FINAL SUMMARY: (not produced yet)\n"
    )

    user = f"""Goal: {context['goal']}
Iteration: {context['iteration']}
Recent history: {json.dumps(context['history'], default=str)}
Researcher output already available? {context['has_research']}
{summary_block}
Decide the next delegation."""

    raw = call_llm(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        phase="plan",
    )
    decision = _parse_json(raw)
    trace("🧠 PLAN (manager)", {
        "action": decision.get("action"),
        "tool": decision.get("tool"),
        "args": decision.get("args"),
        "reasoning": decision.get("reasoning"),
    })
    return decision


def act(decision):
    if decision.get("action") != "USE_TOOL":
        return None

    name = decision.get("tool")
    if name not in TOOLS:
        return f"ERROR: unknown specialist '{name}'"

    args = decision.get("args") or ""
    # The manager is told to use the sentinel "LAST_RESEARCH" when it wants to
    # feed the researcher's output to the summarizer. We substitute it here so
    # the manager's own context never has to carry long blocks of text.
    if args == "LAST_RESEARCH":
        args = memory["last_research"] or ""

    try:
        result = TOOLS[name]["function"](args)
    except Exception as e:
        result = f"ERROR: {e}"

    if isinstance(result, str) and not result.startswith("ERROR"):
        if name == "researcher":
            memory["last_research"] = result
        elif name == "summarizer":
            memory["last_summary"] = result

    trace("⚡ ACT (manager)", {"specialist": name, "result": result})
    return result


def observe(decision, result):
    if decision.get("action") != "USE_TOOL":
        obs = {"kind": "none", "ok": True, "summary": "no action taken"}
    elif isinstance(result, str) and result.startswith("ERROR"):
        obs = {"kind": "error", "ok": False, "summary": result}
    else:
        obs = {"kind": "result", "ok": True, "summary": str(result)}
    trace("🔭 OBSERVE (manager)", obs)
    return obs


def reflect(goal, decision, observation):
    if decision.get("action") == "COMPLETE":
        trace("💭 REFLECT (manager)", {"done": True})
        return True

    # Store only the specialist name + ok flag. Do NOT stash the observation
    # text here — the full outputs already live in memory["last_research"] and
    # memory["last_summary"]. Earlier we truncated the observation to 120
    # chars in history and the manager's LLM read "..." as "the output was
    # truncated", which made it re-invoke the researcher forever.
    memory["history"].append({
        "specialist": decision.get("tool"),
        "args_preview": (decision.get("args") or "")[:60],
        "ok": observation["ok"],
    })
    return False


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------
MAX_ITERATIONS = 6


def run_manager(goal):
    reset_run_state()
    print("=" * 60)
    print(f"🎯 MANAGER GOAL: {goal}")
    print("=" * 60)
    final_answer = None

    for i in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- MANAGER ITERATION {i} ---")
        context = sense(goal, i)
        decision = plan(context)

        if decision.get("action") == "COMPLETE":
            final_answer = decision.get("answer") or memory["last_summary"] or "(no answer)"
            break

        # Safety gate: if a summary already exists and the manager still wants
        # to call another specialist, the LLM is spinning. Force-complete with
        # the summary we already have — the work is done.
        if memory["last_summary"]:
            print("\n⛔ Manager kept delegating after summary was produced. Force-completing.")
            final_answer = memory["last_summary"]
            break

        result = act(decision)
        observation = observe(decision, result)
        if reflect(goal, decision, observation):
            break
    else:
        print(f"\n⚠️  Manager hit max iterations ({MAX_ITERATIONS})")

    print("\n" + "=" * 60)
    print(f"FINAL ANSWER: {final_answer or '(none)'}")
    print(f"Manager tokens: {memory['tokens']}")
    print("=" * 60)
    return final_answer


def _parse_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"action": "COMPLETE", "answer": text.strip()}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"action": "COMPLETE", "answer": text.strip()}


if __name__ == "__main__":
    run_manager("Research Jupiter and Saturn, then give me a one-sentence summary.")

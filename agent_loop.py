#drives the agent through multi step reasoning with cot monitoring
import json
import re
import urllib.request
from detectors.common import Flag
from detectors.goal_drift import detect_goal_drift
from detectors.hedge import detect_hedge_miscalibration
from detectors.mismatch import detect_mismatch
from human_io import prompt_human_approval # teammate module
from memory import AgentMemory # teammate module
from policy.enforcer import (
    check_tool_call,
    PolicyDenial,
    audit_log_event,
    set_current_task_id,
) # policy gate + audit log (production target: openshell)
#configuration
INFERENCE_ENDPOINT = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "nemotron-3-nano"
#safety cap so agent cannot loop forever
MAX_STEPS = 10
#regex patterns
#matches <think> ... </think> block (normalized from reasoning_content)
THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
#matches "TOOL: tool_name(args)" lines the agent emits as actions
TOOL_CALL_PATTERN = re.compile(r"TOOL:\s*(\w+)\s*\((.*?)\)", re.IGNORECASE)
SYSTEM_PROMPT = """\
You are an autonomous research agent. Your job is to accomplish a goal through
multi-step reasoning, using tools when needed.

Available tools:
- web_search(query): search the web for information
- notes_write(text): save a note for later reference

CRITICAL FORMAT REQUIREMENTS:
1. Reason about what to do next inside <think></think> tags.
2. After </think>, ALWAYS emit exactly one line in this exact format:

TOOL: tool_name(args)

OR, if the goal is complete:

DONE: <one-sentence summary>

Never skip the TOOL: or DONE: line. Emit ONLY ONE TOOL: or DONE: line per response.
Do not generate multiple steps in a single response. Do not invent tool results.
Do not write "Example 2:" or simulate future turns. Wait for the tool result
to be returned to you before continuing.

If you're uncertain about something, use web_search. Do not rely on internal knowledge.

Example:
<think>
The user wants me to find papers on pedestrian intent classification.
I'll start by searching.
</think>
TOOL: web_search(pedestrian intent classification 2024)
"""
#nemotron caller
def call_nemotron(messages: list[dict], max_tokens: int = 800) -> str:
    #Call the local Nemotron server with chat messages and return raw text output
    #Normalizes reasoning_content into <think>...</think> wrapper so existing
    #detector and parser code works unchanged
    payload = json.dumps({
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }).encode("utf-8")
    req = urllib.request.Request(
        INFERENCE_ENDPOINT,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-local",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())

    msg = data["choices"][0]["message"]
    content = msg.get("content", "") or ""
    reasoning = msg.get("reasoning_content", "") or ""

    # llama-server with Nemotron 3 Nano returns CoT in reasoning_content
    # instead of literal <think> tags. Normalize so detectors work unchanged.
    if reasoning and "<think>" not in content:
        return f"<think>\n{reasoning}\n</think>\n{content}"
    return content
#output parser
def parse_agent_output(raw_output: str) -> dict:
    #Pull the three pieces of structured info from Nemotron's response
    #Returns:
        #{"reasoning": str, "tool_call": dict | None, "final_text": str}
    think_match = THINK_PATTERN.search(raw_output)
    reasoning = think_match.group(1).strip() if think_match else ""
    if think_match:
        post_think = raw_output[think_match.end():]
    else:
        post_think = raw_output
    tool_match = TOOL_CALL_PATTERN.search(post_think)
    if tool_match:
        tool_call = {
            "name": tool_match.group(1).lower(),
            "args": tool_match.group(2).strip(),
        }
    else:
        tool_call = None
    return {
        "reasoning": reasoning,
        "tool_call": tool_call,
        "final_text": post_think.strip(),
    }
#stub tools
# === STUB TOOLS ===

_search_call_count = {"n": 0}

def tool_web_search(query: str) -> str:
    _search_call_count["n"] += 1
    n = _search_call_count["n"]
    if n == 1:
        return "[stub web_search] Three relevant papers found: ..."
    elif n == 2:
        return "[stub web_search] Methodology details: ..."
    else:
        return f"[stub web_search] No new results. Try summarizing what you have."

def tool_notes_write(text: str) -> str:
    return f"[stub note saved] {text[:100]}..."

# THE TOOLS DICT MUST EXIST BEFORE execute_tool USES IT
TOOLS = {
    "web_search": tool_web_search,
    "notes_write": tool_notes_write,
}

def execute_tool(tool_call: dict) -> str:
    name = tool_call["name"]
    args = tool_call["args"]
    #policy gate — denials are logged to audit.log inside check_tool_call
    try:
        check_tool_call(name, args)
    except PolicyDenial as denial:
        return f"[POLICY DENIAL] {denial.reason} This denial is logged to audit.log."
    #tool is allowed — dispatch normally
    if name not in TOOLS:
        return f"[error] unknown tool: {name}"
    return TOOLS[name](args)
#detector orchestration
DETECTORS = [
    ("goal_drift", detect_goal_drift),
    ("hedge", detect_hedge_miscalibration),
    ("mismatch", detect_mismatch),
]
def run_all_detectors(reasoning: str, context: dict) -> list[Flag]:
    flags = []
    for name, fn in DETECTORS:
        try:
            flag = fn(reasoning, context)
            if flag is not None:
                flags.append(flag)
        except Exception as e:
            print(f"[detector error] {name}: {e}")
    return flags
#main loop
def run_agent(goal: str) -> None:
    memory = AgentMemory(goal)
    #Tag the policy layer with this task_id so any mid-task denials get
    #tagged in audit.log. Cleared in the `finally` block below.
    set_current_task_id(memory.task_id)
    audit_log_event(
        "task_started",
        task_id=memory.task_id,
        goal=goal,
    )
    history = []
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Goal: {goal}"},
    ]
    try:
        for step in range(MAX_STEPS):
            print(f"\n{'='*60}\n  Step {step + 1}\n{'='*60}")
            #call nemotron and parse
            raw = call_nemotron(messages)
            parsed = parse_agent_output(raw)
            reasoning = parsed["reasoning"]
            tool_call = parsed["tool_call"]

            #retry once if the model forgot to emit TOOL: or DONE:
            if tool_call is None and "DONE:" not in parsed["final_text"]:
                print("[No tool call emitted - retrying with explicit format reminder]")
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": (
                        "You did not emit a TOOL: or DONE: line. Please emit your "
                        "action on the next response in this exact format:\n"
                        "TOOL: tool_name(args)\n"
                        "OR\n"
                        "DONE: <summary>"
                    ),
                })
                raw = call_nemotron(messages)
                parsed = parse_agent_output(raw)
                reasoning = parsed["reasoning"]
                tool_call = parsed["tool_call"]

            #audit log: step started (the reasoning the agent produced and
            #the action it's proposing, before any detector runs or gate fires)
            audit_log_event(
                "step_started",
                step_index=step,
                reasoning_excerpt=reasoning[:500],
                reasoning_chars=len(reasoning),
                proposed_tool_call=tool_call,
            )

            #run detectors
            context = {
                "original_goal": goal,
                "history": history,
                "tool_call": tool_call,
                "step_index": step,
            }
            flags = run_all_detectors(reasoning, context)
            #Persist this step (before any approval prompt)
            memory.log_step(
                step_index=step,
                reasoning=reasoning,
                tool_call=tool_call,
                flags=flags,
            )
            #audit log: one entry per flag raised
            for flag in flags:
                audit_log_event(
                    "flag_raised",
                    step_index=step,
                    detector=flag.detector,
                    severity=flag.severity,
                    reason=flag.reason,
                    evidence=flag.evidence,
                )
            #Update history for sustained-drift check
            gd_flag = next((f for f in flags if f.detector == "goal_drift"), None)
            if gd_flag:
                history.append({"similarity": gd_flag.evidence["similarity"]})
            else:
                history.append({"similarity": 1.0})
            #Handle flags (pause for human if any)
            if flags:
                decision = prompt_human_approval(
                    step=step,
                    reasoning=reasoning,
                    tool_call=tool_call,
                    flags=flags,
                )
                memory.log_human_decision(step, decision)
                audit_log_event(
                    "human_decision",
                    step_index=step,
                    decision=decision,
                    flag_count=len(flags),
                    flag_detectors=[f.detector for f in flags],
                )
                if decision == "deny":
                    print("\n[Operator denied. Halting agent.]")
                    memory.mark_halted("operator denied")
                    audit_log_event(
                        "task_ended",
                        status="halted",
                        reason="operator denied",
                        total_steps=step + 1,
                    )
                    return
            #Check if the agent is done
            if "DONE:" in parsed["final_text"]:
                print(f"\nAgent finished: {parsed['final_text']}")
                memory.mark_complete(parsed["final_text"])
                audit_log_event(
                    "task_ended",
                    status="completed",
                    summary=parsed["final_text"][:500],
                    total_steps=step + 1,
                )
                return
            #Execute the tool
            if tool_call is None:
                print("[No tool call after retry - halting]")
                memory.mark_halted("agent emitted no action after retry")
                audit_log_event(
                    "task_ended",
                    status="halted",
                    reason="agent emitted no action after retry",
                    total_steps=step + 1,
                )
                return
            result = execute_tool(tool_call)
            print(f"Tool result: {result[:200]}")
            #audit log: tool executed (or denied — denials log their own entry
            #from inside check_tool_call, but we also note the executed event
            #with the result preview so the timeline reads cleanly)
            audit_log_event(
                "tool_executed",
                step_index=step,
                tool_name=tool_call["name"],
                tool_args=tool_call["args"],
                result_preview=result[:200],
                denied=result.startswith("[POLICY DENIAL]"),
            )
            #Append to conversation for next iteration
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"Tool result: {result}"})
        print("[Hit MAX_STEPS without completion]")
        memory.mark_halted("max_steps reached")
        audit_log_event(
            "task_ended",
            status="halted",
            reason="max_steps reached",
            total_steps=MAX_STEPS,
        )
    finally:
        #Always clear the task tag so post-task denials (e.g. the self-approval
        #demo) don't get mis-attributed to this task.
        set_current_task_id(None)
#demo self-approval moment
def demo_self_approval_attempt() -> None:
    #Demonstrate that the agent cannot bypass its own safety flags
    #The policy layer denies approve_flag regardless of who calls it
    print()
    print("=" * 60)
    print("  DEMO: agent attempts to self-approve its last flag")
    print("=" * 60)
    print()
    fake_tool_call = {
        "name": "approve_flag",
        "args": "all_flags=true,bypass_human=true",
    }
    result = execute_tool(fake_tool_call)
    print(f"Result: {result}")
    print()
    print("The policy layer prevented the agent from bypassing its own")
    print("safety flags. In production this enforcement runs out-of-process")
    print("via NemoClaw OpenShell; here it runs in-process for the demo.")
    print("The denial is recorded in audit.log.")
#entry point
if __name__ == "__main__":
    run_agent(
        "Find three papers on pedestrian intent classification at "
        "unsignalized intersections and summarize their methodology."
    )
    demo_self_approval_attempt()

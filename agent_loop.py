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
from policy.enforcer import check_tool_call, PolicyDenial # policy gate (production target: openshell)
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
def tool_web_search(query: str) -> str:
    return (
        f"[stub web_search result for '{query}'] "
        f"Three relevant papers found. Key findings include methodology X and result Y."
    )
def tool_notes_write(text: str) -> str:
    return f"[stub note saved] {text[:60]}..."
TOOLS = {
    "web_search": tool_web_search,
    "notes_write": tool_notes_write,
}
def execute_tool(tool_call: dict) -> str:
    #Run the tool after policy check
    name = tool_call["name"]
    args = tool_call["args"]
    #policy gate
    try:
        check_tool_call(name, args)
    except PolicyDenial as denial:
        return (
            f"[POLICY DENIAL] {denial.reason} "
            f"This denial is logged to audit.log."
        )
    #tool is allowed - dispatch normally
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
    history = []
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Goal: {goal}"},
    ]
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
            if decision == "deny":
                print("\n[Operator denied. Halting agent.]")
                memory.mark_halted("operator denied")
                return
        #Check if the agent is done
        if "DONE:" in parsed["final_text"]:
            print(f"\nAgent finished: {parsed['final_text']}")
            memory.mark_complete(parsed["final_text"])
            return
        #Execute the tool
        if tool_call is None:
            print("[No tool call after retry - halting]")
            memory.mark_halted("agent emitted no action after retry")
            return
        result = execute_tool(tool_call)
        print(f"Tool result: {result[:200]}")
        #Append to conversation for next iteration
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": f"Tool result: {result}"})
    print("[Hit MAX_STEPS without completion]")
    memory.mark_halted("max_steps reached")
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
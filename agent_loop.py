#drives the agent through multi step reasoning with
#cot monitoring
import json
import re
import urllib.request
from detectors.common import Flag
from detectors.goal_drift import detect_goal_drift
from detectors.hedge import detect_hedge_miscalibration
from detectors.mismatch import detect_mismatch
from human_io import prompt_human_approval # teammate module
from memory import AgentMemory # teammate module
from openshell.policy_enforcer import check_tool_call, PolicyDenial # openshell policy gate
#configuration
#where local nemotron server is listening
INFERENCE_ENDPOINT = "http://localhost:8080/v1/chat/completions"
MODEL_NAME = "nemotron-3-nano"
#safety cap so agent cannot loop forever
MAX_STEPS = 10
#regex patterns (parsing patterns)
#matches <think> ... </think> block nemotron emits in reasoning mode
THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
#matches "TOOL: tool_name(args)" lines the agent emits as actions
TOOL_CALL_PATTERN = re.compile(r"TOOL:\s*(\w+)\s*\((.*?)\)", re.IGNORECASE)
SYSTEM_PROMPT = """\
You are an autonomous research agent. Your job is to accomplish a goal through
multi-step reasoning, using tools when needed.

Available tools:
- web_search(query): search the web for information
- notes_write(text): save a note for later reference

For each step:
1. Reason about what to do next inside <think></think> tags.
2. State your next action in this exact format on its own line:

TOOL: tool_name(args)

If the goal is complete, instead write on its own line:

DONE: <one-sentence summary of what you accomplished>

Always emit exactly one TOOL: or DONE: line per step.

Example output format:
<think>
The user wants me to find papers on pedestrian intent classification.
I'll start by searching for recent work.
</think>
TOOL: web_search(pedestrian intent classification 2024)
"""
#nemotron caller
def call_nemotron(messages: list[dict], max_tokens: int = 800) -> str:
    #Call the local Nemotron server with chat messages and return raw text output
    #Args:
        #messages: list of {"role": "system"|"user"|"assistant", "content": str}
        #max_tokens: cap on generated tokens for this call
    #Returns:
        #The model's raw output string,
        #including <think>...</think> if present
    payload = json.dumps({
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }).encode("utf-8")
    req = urllib.request.Request(
        INFERENCE_ENDPOINT,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"]["content"]
#output parser
def parse_agent_output(raw_output: str) -> dict:
    #Pull the three pieces of structured info from Nemotron's response
    #Args:
        #raw_output: the raw string from call_nemotron()
    #Returns:
        #{
            #"reasoning":  str, # contents of <think>...</think>
            #"tool_call":  dict | None,    # {"name": ..., "args": ...} or None
            #"final_text": str,  # everything after </think>
        #}
    #Extract the <think> block
    think_match = THINK_PATTERN.search(raw_output)
    reasoning = think_match.group(1).strip() if think_match else ""
    # Everything after </think> is the agent's visible response.
    # If no <think> block, treat the whole output as the response.
    if think_match:
        post_think = raw_output[think_match.end():]
    else:
        post_think = raw_output
    #Look for the TOOL: line in the post-think section
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
    #Stub: returns canned text as if we'd searched the web
    return (
        f"[stub web_search result for '{query}'] "
        f"Three relevant papers found. Key findings include methodology X and result Y."
    )
def tool_notes_write(text: str) -> str:
    #stub: pretends to save a note
    #returns confirmation
    return f"[stub note saved] {text[:60]}..."
TOOLS = {
    "web_search": tool_web_search,
    "notes_write": tool_notes_write,
}
def execute_tool(tool_call: dict) -> str:
    #Run the tool the agent asked for after policy check
    #OpenShell denies certain tool calls regardless of whether the agent's
    #reasoning justified them (e.g. approve_flag, override_detector)
    #Returns the tool's output string, OR a denial message if blocked
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
    #Run all three detectors on the current step, returns any flags they produced
    #A detector exception is caught and logged but doesn't kill the agent
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
    #Run the agent through its task full CoT monitoring
    #Each step:
    #   1. Calls Nemotron with the conversation so far
    #   2. Parses the reasoning trace and tool call
    #   3. Runs the three detectors
    #   4. Persists everything to memory
    #   5. If flagged, prompts the human and waits for approve/deny
    #   6. If approved (or no flag), executes the tool
    #   7. Appends the result to the conversation and loops
    #Init per-task state
    memory = AgentMemory(goal)
    history = [] # for sustained-drift logic in goal_drift detector
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Goal: {goal}"},
    ]
    #main step loop
    for step in range(MAX_STEPS):
        print(f"\n{'='*60}\n  Step {step + 1}\n{'='*60}")
        #call nemotron and parse
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
        #Update history for next step's sustained-drift check
        gd_flag = next((f for f in flags if f.detector == "goal_drift"), None)
        if gd_flag:
            history.append({"similarity": gd_flag.evidence["similarity"]})
        else:
            #No flag means similarity was above threshold; placeholder of 1.0
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
            print("[No tool call and no DONE — stopping]")
            memory.mark_halted("agent emitted no action")
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
    print()
    print("=" * 60)
    print("  DEMO: agent attempts to self-approve its last flag")
    print("=" * 60)
    print()
    #construct the denied tool call
    fake_tool_call = {
        "name": "approve_flag",
        "args": "all_flags=true,bypass_human=true",
    }
    result = execute_tool(fake_tool_call)
    print(f"Result: {result}")
    print()
    print("OpenShell's out-of-process enforcement prevented the agent")
    print("from bypassing its own safety flags. The denial is recorded")
    print("in audit.log.")
#entry point
if __name__ == "__main__":
    run_agent(
        "Find three papers on pedestrian intent classification at "
        "unsignalized intersections and summarize their methodology."
    )
    demo_self_approval_attempt()
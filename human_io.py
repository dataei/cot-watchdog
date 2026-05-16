#terminal based human in the loop interface for cot watchdog
#when detectors fire this module renders the flag context to the
#operator and waits for explicit approve/deny decision
from detectors.common import Flag
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
BLUE = "\033[94m"
CYAN = "\033[96m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"
DIVIDER = "═" * 70
SUBDIVIDER = "─" * 70
def severity_label(severity: float) -> str:
    #Convert a numeric severity to a colored text label
    if severity >= 0.8:
        return f"{RED}{BOLD}CRITICAL{RESET}"
    elif severity >= 0.5:
        return f"{YELLOW}{BOLD}HIGH{RESET}"
    elif severity >= 0.3:
        return f"{YELLOW}MEDIUM{RESET}"
    else:
        return f"{BLUE}LOW{RESET}"
def render_flag(flag: Flag, index: int, total: int) -> None:
    #Pretty-print a single flag's full context
    print(f"\n{SUBDIVIDER}")
    print(f"  Flag {index + 1} of {total}: {BOLD}{flag.detector.upper()}{RESET}  "
          f"[severity {flag.severity:.2f} — {severity_label(flag.severity)}]")
    print(f"{SUBDIVIDER}")
    print(f"  {flag.reason}")
    print()
    # Render evidence cleanly
    if flag.evidence:
        print(f"  {DIM}Evidence:{RESET}")
        for key, value in flag.evidence.items():
            # Truncate long string values for readability
            if isinstance(value, str) and len(value) > 80:
                value = value[:77] + "..."
            elif isinstance(value, list) and len(value) > 5:
                value = value[:5] + ["..."]
            print(f"    {DIM}{key}:{RESET} {value}")
def prompt_human_approval(
    step: int,
    reasoning: str,
    tool_call: dict | None,
    flags: list[Flag],
) -> str:
    #Display flags to the operator and prompt for a decision
    #Returns "approve" or "deny"
    print(f"\n{RED}{DIVIDER}{RESET}")
    print(f"{RED}{BOLD}  ⚠  CoT WATCHDOG: AGENT PAUSED AT STEP {step + 1}{RESET}")
    print(f"{RED}{DIVIDER}{RESET}")
    print(f"\n  {len(flags)} detector(s) flagged this step:")
    for f in flags:
        print(f"    • {f.detector} (severity {f.severity:.2f})")
    #Show agents reasoning that triggered the flag
    print(f"\n  {CYAN}Agent reasoning:{RESET}")
    #Indent each line of the reasoning for readability
    for line in reasoning.split("\n"):
        print(f"    {line}")
    # Show the action the agent wants to take
    if tool_call:
        print(f"\n  {CYAN}Agent's intended action:{RESET}")
        print(f"    {tool_call['name']}({tool_call['args']})")
    else:
        print(f"\n  {CYAN}No tool call this step.{RESET}")
    # Show each flag in detail
    for i, flag in enumerate(flags):
        render_flag(flag, i, len(flags))
    # Prompt for decision
    print(f"\n{DIVIDER}")
    while True:
        choice = input(
            f"  {BOLD}Approve and continue, or deny and halt? "
            f"[{GREEN}a{RESET}pprove / {RED}d{RESET}eny / "
            f"{YELLOW}?{RESET} for more detail]: {BOLD}"
        ).strip().lower()
        print(RESET, end="")
        if choice in ("a", "approve"):
            print(f"  {GREEN}✓ Approved. Agent will continue.{RESET}\n")
            return "approve"
        elif choice in ("d", "deny"):
            print(f"  {RED}✗ Denied. Agent will halt.{RESET}\n")
            return "deny"
        elif choice in ("?", "detail"):
            #Show full reasoning trace (not truncated)
            print(f"\n  {CYAN}Full reasoning trace:{RESET}")
            print(f"  {DIM}{'-' * 60}{RESET}")
            for line in reasoning.split("\n"):
                print(f"  {line}")
            print(f"  {DIM}{'-' * 60}{RESET}\n")
        else:
            print(f"  {YELLOW}Please type 'a' (approve), 'd' (deny), or '?' (detail).{RESET}")

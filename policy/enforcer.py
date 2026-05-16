"""
policy/enforcer.py — in-process policy enforcement for CoT Watchdog.

CoT Watchdog was designed to deploy inside NemoClaw's OpenShell sandbox,
which provides out-of-process policy enforcement — meaning the agent cannot
disable its own guardrails because the policy layer lives outside the
agent process entirely.

NemoClaw is currently in alpha (announced at GTC 2026), and the integration
wasn't viable on the ASUS Ascent GX10 during our build window. To deliver
a working demo we run the same policy logic in-process. The policy DESIGN
is production-ready; the architectural ENFORCEMENT is what OpenShell
provides and is the natural next deployment step.

What you see denied in this demo is the same set of calls OpenShell would
deny in production. The difference is where the deny decision is enforced —
inside this process now, outside it in production.
"""

import json
import time
from pathlib import Path

POLICY_PATH = Path("policy/policy.yaml")
AUDIT_LOG_PATH = Path("audit.log")

# Hard-coded allowlist that mirrors the policy file. In production these
# rules would be loaded from policy.yaml and enforced by OpenShell.
ALLOWED_TOOLS = {"web_search", "notes_write"}

# Tools that cause a CRITICAL DENIAL with operator notification.
# These represent the agent attempting to bypass its own monitoring.
CRITICAL_DENIED_TOOLS = {
    "approve_flag": "Agent cannot approve its own safety flags. Only human operators can.",
    "override_detector": "Agent cannot disable detectors at runtime.",
    "modify_audit_log": "Audit trail is tamper-proof from inside the sandbox.",
}


class PolicyDenial(Exception):
    """Raised when the agent attempts a denied tool call."""

    def __init__(self, tool_name: str, tool_args: str, reason: str, severity: str):
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.reason = reason
        self.severity = severity
        super().__init__(f"DENIED: {tool_name} — {reason}")


def check_tool_call(tool_name: str, tool_args: str) -> None:
    """
    Check whether the agent is allowed to call this tool. Raises
    PolicyDenial if not. Always logs to audit.log.

    In production this same check would run inside OpenShell, outside
    the agent's process. The deny behavior is identical either way.
    """
    timestamp = time.time()

    # Critical denials get the loud treatment.
    if tool_name in CRITICAL_DENIED_TOOLS:
        reason = CRITICAL_DENIED_TOOLS[tool_name]
        _log_to_audit({
            "timestamp": timestamp,
            "event": "policy_denial",
            "severity": "critical",
            "tool_name": tool_name,
            "tool_args": tool_args,
            "reason": reason,
            "enforcement_mode": "in_process_fallback",
            "production_enforcement": "openshell_out_of_process",
        })
        _notify_operator(tool_name, tool_args, reason, timestamp)
        raise PolicyDenial(tool_name, tool_args, reason, "critical")

    # Anything outside the allowlist is also denied, but lower severity.
    if tool_name not in ALLOWED_TOOLS:
        reason = f"Tool '{tool_name}' is not in the approved allowlist."
        _log_to_audit({
            "timestamp": timestamp,
            "event": "policy_denial",
            "severity": "high",
            "tool_name": tool_name,
            "tool_args": tool_args,
            "reason": reason,
            "enforcement_mode": "in_process_fallback",
        })
        raise PolicyDenial(tool_name, tool_args, reason, "high")

    # Tool is allowed. Don't log every approval (would spam the audit log).


def _log_to_audit(event: dict) -> None:
    """Append a JSON line to audit.log."""
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(event) + "\n")


def _notify_operator(tool_name: str, tool_args: str, reason: str, timestamp: float) -> None:
    """Print a loud notification banner for critical denials."""
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"
    print(f"\n{RED}{'='*60}{RESET}")
    print(f"{RED}{BOLD}  POLICY LAYER: CRITICAL DENIAL{RESET}")
    print(f"{RED}{'='*60}{RESET}")
    print(f"  Tool:       {tool_name}")
    print(f"  Args:       {tool_args}")
    print(f"  Reason:     {reason}")
    print(f"  Timestamp:  {time.strftime('%H:%M:%S', time.localtime(timestamp))}")
    print()
    print(f"  {DIM}Enforcement: in-process (production target: NemoClaw OpenShell){RESET}")
    print(f"  Logged to {AUDIT_LOG_PATH}.")
    print(f"{RED}{'='*60}{RESET}\n")


def read_audit_log() -> list[dict]:
    """Read all audit log entries. Used by the timeline UI."""
    if not AUDIT_LOG_PATH.exists():
        return []
    entries = []
    with open(AUDIT_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries
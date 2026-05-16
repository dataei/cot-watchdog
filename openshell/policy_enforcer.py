#fallback policy enforcement when real openshell
#is not fully config
import json
import time
from pathlib import Path
 
POLICY_PATH = Path("openshell/policy.yaml")
AUDIT_LOG_PATH = Path("audit.log")
ALLOWED_TOOLS = {"web_search", "notes_write"}
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
 
    Args:
        tool_name: the tool the agent is trying to call
        tool_args: the arguments it's trying to pass
 
    Raises:
        PolicyDenial: if the tool is on the denial list
    """
    timestamp = time.time()
    if tool_name in CRITICAL_DENIED_TOOLS:
        reason = CRITICAL_DENIED_TOOLS[tool_name]
        _log_to_audit({
            "timestamp": timestamp,
            "event": "policy_denial",
            "severity": "critical",
            "tool_name": tool_name,
            "tool_args": tool_args,
            "reason": reason,
        })
        _notify_operator(tool_name, tool_args, reason, timestamp)
        raise PolicyDenial(tool_name, tool_args, reason, "critical")
    if tool_name not in ALLOWED_TOOLS:
        reason = f"Tool '{tool_name}' is not in the approved allowlist."
        _log_to_audit({
            "timestamp": timestamp,
            "event": "policy_denial",
            "severity": "high",
            "tool_name": tool_name,
            "tool_args": tool_args,
            "reason": reason,
        })
        raise PolicyDenial(tool_name, tool_args, reason, "high")
def _log_to_audit(event: dict) -> None:
    """Append a JSON line to audit.log."""
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(event) + "\n")
def _notify_operator(tool_name: str, tool_args: str, reason: str, timestamp: float) -> None:
    """Print a loud notification banner for critical denials."""
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    print(f"\n{RED}{'='*60}{RESET}")
    print(f"{RED}{BOLD}  OPENSHELL: CRITICAL DENIAL{RESET}")
    print(f"{RED}{'='*60}{RESET}")
    print(f"  Tool:       {tool_name}")
    print(f"  Args:       {tool_args}")
    print(f"  Reason:     {reason}")
    print(f"  Timestamp:  {time.strftime('%H:%M:%S', time.localtime(timestamp))}")
    print()
    print(f"  This denial has been logged to {AUDIT_LOG_PATH}.")
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
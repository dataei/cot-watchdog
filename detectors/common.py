from dataclasses import dataclass, field
from typing import Optional, Any

@dataclass
class Flag:
    #a single flag emittted by detector
    detector: str # which detector fired: "goal_drift" | "mismatch" | "hedge"
    severity: float # 0.0 = barely a concern, 1.0 = stop immediately
    reason: str # one-sentence human-readable explanation
    evidence: dict = field(default_factory=dict)  # raw data for the operator
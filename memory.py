#persistent storage for cot watchdog audit trail
#each agent run gets a unique task_id
#all reasoning steps, tool calls, flags, and human decisions
#are persisted to a local SQLite database
#for post-hoc audit of the full reasoning chain
import json
import sqlite3
import time
import uuid
from pathlib import Path
from detectors.common import Flag
#config
# Default location of the SQLite database, one database can hold many tasks
DB_PATH = Path("watchdog_memory.db")
#schema
#executed on every AgentMemory instantiation (CREATE TABLE IF NOT EXISTS)
#so it's safe to run repeatedly and safe to share one DB across runs
SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id        TEXT PRIMARY KEY,
    goal           TEXT NOT NULL,
    started_at     REAL NOT NULL,
    completed_at   REAL,
    final_status   TEXT,         -- "completed" | "halted" | "max_steps" | NULL while running
    final_summary  TEXT
);
 
CREATE TABLE IF NOT EXISTS steps (
    step_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id     TEXT NOT NULL,
    step_index  INTEGER NOT NULL,
    reasoning   TEXT NOT NULL,
    tool_name   TEXT,             -- NULL if the step made no tool call
    tool_args   TEXT,             -- NULL if the step made no tool call
    timestamp   REAL NOT NULL,
    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
);
 
CREATE TABLE IF NOT EXISTS flags (
    flag_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id          TEXT NOT NULL,
    step_index       INTEGER NOT NULL,
    detector         TEXT NOT NULL,    -- "goal_drift" | "hedge" | "mismatch"
    severity         REAL NOT NULL,
    reason           TEXT NOT NULL,
    evidence_json    TEXT NOT NULL,    -- the Flag.evidence dict, JSON-serialized
    human_decision   TEXT,             -- "approve" | "deny" | NULL while waiting
    timestamp        REAL NOT NULL,
    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
);
 
CREATE INDEX IF NOT EXISTS idx_steps_task ON steps(task_id, step_index);
CREATE INDEX IF NOT EXISTS idx_flags_task ON flags(task_id, step_index);
"""
#main class
class AgentMemory:
    #manages persistent storage for a single agent run
    #a new task_id is generated on instantiation, all subsequent logging
    #methods write to that task_id automatically, so the agent loop never
    #has to track the ID directly
    def __init__(self, goal: str, db_path: Path = DB_PATH):
        #Start a new task and initialize the database if needed.
 
        #Args:
        #     goal: the human-provided goal string for this agent run
        #     db_path: where to store the SQLite database
        self.db_path = Path(db_path)
        self.task_id = str(uuid.uuid4())[:8]  # short, readable, unique-enough
        self.goal = goal
        # Ensure schema exists, then insert the new task row
        with self._conn() as conn:
            conn.executescript(SCHEMA)
            conn.execute(
                "INSERT INTO tasks (task_id, goal, started_at) VALUES (?, ?, ?)",
                (self.task_id, goal, time.time()),
            )
        print(f"[memory] task started: {self.task_id}")
    def _conn(self) -> sqlite3.Connection:
        #Open a new SQLite connection, caller is responsible for closing
        #(use as a context manager)
        return sqlite3.connect(self.db_path)
    # logging methods (called by the agent loop)
    def log_step(
        self,
        step_index: int,
        reasoning: str,
        tool_call: dict | None,
        flags: list[Flag],
    ) -> None:
        
        # Record everything about a single agent step: the reasoning trace,
        # the tool the agent wants to call, and any flags detectors raised
        # Args:
        #     step_index: zero-indexed step number within this task
        #     reasoning: the agent's <think>...</think> content for this step
        #     tool_call: {"name": ..., "args": ...} dict, or None if no tool call
        #     flags: list of Flag objects detectors produced (empty if clean)
        now = time.time()
        tool_name = tool_call["name"] if tool_call else None
        tool_args = tool_call["args"] if tool_call else None
        with self._conn() as conn:
            #Insert the step row
            conn.execute(
                """INSERT INTO steps
                   (task_id, step_index, reasoning, tool_name, tool_args, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (self.task_id, step_index, reasoning, tool_name, tool_args, now),
            )
            # Insert one row per flag (if any)
            for flag in flags:
                conn.execute(
                    """INSERT INTO flags
                       (task_id, step_index, detector, severity, reason,
                        evidence_json, timestamp)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        self.task_id,
                        step_index,
                        flag.detector,
                        flag.severity,
                        flag.reason,
                        # default=str handles non-JSON-native types (numpy floats, etc.)
                        json.dumps(flag.evidence, default=str),
                        now,
                    ),
                )
    def log_human_decision(self, step_index: int, decision: str) -> None:
        # Update all flags at the given step with the human's decision
        # Called after prompt_human_approval returns "approve" or "deny"
        # Args:
        #     step_index: the step the human just decided on
        #     decision: "approve" or "deny"
        if decision not in ("approve", "deny"):
            raise ValueError(f"decision must be 'approve' or 'deny', got {decision!r}")
 
        with self._conn() as conn:
            conn.execute(
                """UPDATE flags
                   SET human_decision = ?
                   WHERE task_id = ? AND step_index = ?""",
                (decision, self.task_id, step_index),
            )
 
    # finalizers (mark task as ended)
    def mark_complete(self, summary: str) -> None:
        # Mark the task as completed. Called when the agent emits DONE:.
 
        # Args:
        #     summary: the agent's final summary string
        with self._conn() as conn:
            conn.execute(
                """UPDATE tasks
                   SET completed_at = ?,
                       final_status = 'completed',
                       final_summary = ?
                   WHERE task_id = ?""",
                (time.time(), summary, self.task_id),
            )
        print(f"[memory] task {self.task_id} marked complete")
    def mark_halted(self, reason: str = "operator denied") -> None:
        # Mark the task as halted before completion. Called when the human
        # denies a flag, when the agent emits no action, or when MAX_STEPS hits.
 
        # Args:
        #     reason: human-readable explanation of why the task halted
        with self._conn() as conn:
            conn.execute(
                """UPDATE tasks
                   SET completed_at = ?,
                       final_status = 'halted',
                       final_summary = ?
                   WHERE task_id = ?""",
                (time.time(), reason, self.task_id),
            )
        print(f"[memory] task {self.task_id} marked halted: {reason}")
    #export (for the audit timeline UI)
    def export_timeline(self) -> dict:
        # Pull the full task history as a structured dict, suitable for
        # rendering the post-hoc drift timeline.
 
        # Returns:
        #     {
        #         "task":  {task_id, goal, started_at, completed_at, ...},
        #         "steps": [ {step_index, reasoning, tool_name, tool_args, ...}, ... ],
        #         "flags": [ {step_index, detector, severity, evidence, ...}, ... ],
        #     }
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            task_row = conn.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (self.task_id,)
            ).fetchone()
            step_rows = conn.execute(
                "SELECT * FROM steps WHERE task_id = ? ORDER BY step_index",
                (self.task_id,),
            ).fetchall()
            flag_rows = conn.execute(
                "SELECT * FROM flags WHERE task_id = ? ORDER BY step_index, flag_id",
                (self.task_id,),
            ).fetchall()
        return {
            "task": dict(task_row) if task_row else None,
            "steps": [dict(s) for s in step_rows],
            "flags": [
                # Hydrate the evidence JSON back into a dict for convenience
                {**dict(f), "evidence": json.loads(f["evidence_json"])}
                for f in flag_rows
            ],
        }
#static queries (don't require a live AgentMemory instance)
def list_all_tasks(db_path: Path = DB_PATH) -> list[dict]:
    # Return all tasks in the database, newest first, useful for a "history"
    # view across multiple agent runs
    if not Path(db_path).exists():
        return []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM tasks ORDER BY started_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]
def load_task_timeline(task_id: str, db_path: Path = DB_PATH) -> dict | None:
    # Load a specific task's full timeline by ID, without needing the
    # original AgentMemory instance, used for post-hoc audit views
    # Returns None if the task_id doesn't exist
    if not Path(db_path).exists():
        return None
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        task = conn.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
        ).fetchone()
        if task is None:
            return None
        steps = conn.execute(
            "SELECT * FROM steps WHERE task_id = ? ORDER BY step_index",
            (task_id,),
        ).fetchall()
        flags = conn.execute(
            "SELECT * FROM flags WHERE task_id = ? ORDER BY step_index, flag_id",
            (task_id,),
        ).fetchall()
    return {
        "task": dict(task),
        "steps": [dict(s) for s in steps],
        "flags": [
            {**dict(f), "evidence": json.loads(f["evidence_json"])}
            for f in flags
        ],
    }
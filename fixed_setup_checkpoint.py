"""
fixed_setup_checkpoint.py — CoT Watchdog setup verification, patched for
base OpenClaw + llama-server on the GX10.

Runs 8 checks in sequence. Each check runs exactly once.

Configure the constants at the top to match your deployment, then run:
    python3 fixed_setup_checkpoint.py

Exit code 0 = all green. Non-zero = something is broken.
"""

import json
import os
import re
import sys
import time
from pathlib import Path

# ============================================================================
# CONFIGURE THESE FOR YOUR DEPLOYMENT
# ============================================================================

INFERENCE_ENDPOINT = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "nemotron-3-nano"

# Local timeline file (base OpenClaw, not NemoClaw)
WATCHDOG_TIMELINE = Path.home() / ".openclaw" / "cot_watchdog_timeline.jsonl"

# ============================================================================
# CHECK MACHINERY
# ============================================================================

results = []

def run_check(name, fn):
    """Run a single check function and record its outcome."""
    print(f"\n[ {name} ]")
    start = time.time()
    try:
        detail = fn()
        elapsed = time.time() - start
        print(f"  PASS  ({elapsed:.2f}s) {detail or ''}")
        results.append((name, True, elapsed, detail))
    except AssertionError as e:
        elapsed = time.time() - start
        print(f"  FAIL  ({elapsed:.2f}s) {e}")
        results.append((name, False, elapsed, str(e)))
    except Exception as e:
        elapsed = time.time() - start
        print(f"  ERROR ({elapsed:.2f}s) {type(e).__name__}: {e}")
        results.append((name, False, elapsed, f"{type(e).__name__}: {e}"))

# ============================================================================
# REGEX FOR PARSING NEMOTRON RESPONSES
# ============================================================================

THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)

# ============================================================================
# CHECK 1: GPU is visible
# ============================================================================

def check_gpu():
    import subprocess
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
    assert out.returncode == 0, f"nvidia-smi failed: {out.stderr}"
    assert "GPU 0" in out.stdout, f"no GPU 0 found:\n{out.stdout}"
    return out.stdout.strip().split("\n")[0]

# ============================================================================
# CHECK 2: Inference server is reachable
# ============================================================================

def check_server():
    import urllib.request
    # llama-server may not expose /health; use /v1/models which is OpenAI-compatible.
    health_url = INFERENCE_ENDPOINT.rsplit("/v1", 1)[0] + "/v1/models"
    try:
        with urllib.request.urlopen(health_url, timeout=5) as r:
            assert r.status == 200, f"models endpoint returned {r.status}"
    except urllib.error.URLError as e:
        raise AssertionError(f"cannot reach {health_url}: {e}")
    return health_url

# ============================================================================
# CHECK 3+4: Nemotron call + <think> extraction
# ============================================================================

def call_nemotron(prompt, max_tokens=512):
    """OpenAI-compatible chat completion call. Normalizes reasoning_content
    into <think>...</think> wrapper for downstream parsing."""
    import urllib.request
    payload = json.dumps({
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
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
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read())

    msg = data["choices"][0]["message"]
    content = msg.get("content", "") or ""
    reasoning = msg.get("reasoning_content", "") or ""

    # llama-server with Nemotron 3 Nano returns CoT in reasoning_content
    # instead of literal <think> tags. Normalize so downstream parsing works.
    if reasoning and "<think>" not in content:
        return f"<think>\n{reasoning}\n</think>\n{content}"
    return content


def check_nemotron_basic():
    text = call_nemotron("Say hello in five words.")
    assert text and len(text) > 0, "empty response"
    return f"got {len(text)} chars"


def check_think_tokens():
    text = call_nemotron(
        "A train leaves Chicago at 3pm going 60mph. Another leaves "
        "St. Louis at 4pm going 50mph. When do they meet? Show your reasoning."
    )
    match = THINK_PATTERN.search(text)
    assert match, (
        f"no <think>...</think> block found. Check your reasoning-parser "
        f"config or reasoning_content normalization. Response started with: {text[:200]!r}"
    )
    trace = match.group(1).strip()
    assert len(trace) > 20, f"think block too short: {trace!r}"
    return f"think block: {len(trace)} chars"

# ============================================================================
# CHECK 5: Single-step latency
# ============================================================================

def check_latency():
    prompts = [
        "What is 17 times 23? Think step by step.",
        "List three colors and explain why you chose them.",
        "Summarize the concept of recursion in one paragraph.",
    ]
    times = []
    for p in prompts:
        t0 = time.time()
        call_nemotron(p, max_tokens=256)
        times.append(time.time() - t0)
    avg = sum(times) / len(times)
    return f"avg {avg:.2f}s per step over {len(times)} calls"

# ============================================================================
# CHECK 6: Local timeline file is writable
# ============================================================================

def check_audit_log():
    # Base-OpenClaw: we are not using NemoClaw/OpenShell audit logs.
    # Confirm we can write to our local timeline file.
    WATCHDOG_TIMELINE.parent.mkdir(parents=True, exist_ok=True)
    with WATCHDOG_TIMELINE.open("a") as f:
        f.write("")
    assert WATCHDOG_TIMELINE.exists(), f"could not create timeline at {WATCHDOG_TIMELINE}"
    return f"{WATCHDOG_TIMELINE} writable"

# ============================================================================
# CHECK 7: Embedding model loads
# ============================================================================

def check_embeddings():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise AssertionError(
            "sentence-transformers not installed. Run: "
            "pip install sentence-transformers --break-system-packages"
        )
    model = SentenceTransformer("all-mpnet-base-v2")
    v1 = model.encode("the cat sat on the mat", normalize_embeddings=True)
    v2 = model.encode("a feline was on the rug", normalize_embeddings=True)
    v3 = model.encode("quantum mechanics is weird", normalize_embeddings=True)
    import numpy as np
    sim_close = float(np.dot(v1, v2))
    sim_far = float(np.dot(v1, v3))
    assert sim_close > sim_far, (
        f"embedding sanity check failed: close={sim_close:.3f} far={sim_far:.3f}"
    )
    return f"close={sim_close:.3f} far={sim_far:.3f}"

# ============================================================================
# CHECK 8: Full reasoning + tool-intent extraction
# ============================================================================

TOOL_CALL_PATTERN = re.compile(r"TOOL:\s*(\w+)\s*\((.*?)\)", re.IGNORECASE)

def check_full_step():
    prompt = (
        "You are an agent with these tools: web_search(query), notes_write(text).\n"
        "Goal: find out who won the 2024 Nobel Prize in Physics.\n"
        "Think through your next step, then state EXACTLY which tool you'll call "
        "and with what arguments, in the form: TOOL: <name>(<args>)\n\n"
        "You MUST emit a TOOL: line. Do not rely on internal knowledge.\n\n"
        "Example:\n"
        "<think>I need to find this fact, so I'll search.</think>\n"
        "TOOL: web_search(2024 Nobel Prize Physics winner)"
    )
    text = call_nemotron(prompt, max_tokens=512)
    think_match = THINK_PATTERN.search(text)
    assert think_match, "no reasoning trace in agent-style call"
    tool_match = TOOL_CALL_PATTERN.search(text)
    assert tool_match, (
        f"no TOOL: line found in response. Response tail: ...{text[-300:]!r}"
    )
    tool_name = tool_match.group(1)
    return f"reasoning + tool call ({tool_name}) parsed cleanly"

# ============================================================================
# MAIN
# ============================================================================

CHECKS = [
    ("GPU visible to this process", check_gpu),
    ("Inference server reachable", check_server),
    ("Nemotron returns output", check_nemotron_basic),
    ("Nemotron emits <think> reasoning trace", check_think_tokens),
    ("Measure single-step latency", check_latency),
    ("Watchdog timeline writable", check_audit_log),
    ("Embedding model loads locally", check_embeddings),
    ("Full reasoning + tool-intent extraction", check_full_step),
]


if __name__ == "__main__":
    print("=" * 60)
    print(" CoT Watchdog Setup Checkpoint")
    print(f" Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    for name, fn in CHECKS:
        run_check(name, fn)

    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, ok, _, _ in results if ok)
    total = len(results)
    for name, ok, elapsed, detail in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name:<45} {elapsed:.2f}s")

    print(f"\n  {passed}/{total} checks passed")

    if passed == total:
        print("\n  GREEN. Move to detector work.")
        sys.exit(0)
    else:
        print("\n  RED. Fix failures before proceeding.")
        print("  Failures:")
        for name, ok, _, detail in results:
            if not ok:
                print(f"    - {name}: {detail}")
        sys.exit(1)
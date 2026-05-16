#cot watchdog setup checkpoint
#run once gx10 and monitor are setup. confirms if
#environment is good
#if everything passes we are good to go
#if anything fails, output says what to fix
#execute in terminal 
#python setup_checkpoint.py
#exit code 0 is success
#non zero means something is broken
#configure the four constants below to match 
#deployment before running
import json
import os
import re
import sys
import time
from pathlib import Path

#where our nemotron inference server is running
#all expose OpenAI-compatible endpoints, point it
#to what we use
INFERENCE_ENDPOINT = "http://localhost:8080/v1/chat/completions"
MODEL_NAME = "nemotron-3-nano"
#this is where our openshell writes its audit log
#check nemoclaw docs for exact path in our install
#it is a common default
OPENSHELL_AUDIT_LOG = Path.home() / ".nemoclaw" / "audit.log"
#where openclaw agent stores its working state
OPENCLAW_STATE_DIR = Path.home() / ".openclaw" / "state"
results = []
def check(name):
    #decorator to run a check and take note of pass/fail with timing
    def wrap(fn):
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
        return fn
    return wrap

#check 1 where GPU is visible

@check("GPU visible to this process")
def check_gpu():
    import subprocess
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
    assert out.returncode == 0, f"nvidia-smi failed: {out.stderr}"
    assert "GPU 0" in out.stdout, f"no GPU 0 found:\n{out.stdout}"
    # GX10 has Blackwell GB10. If we see something else, we're not on the right machine.
    return out.stdout.strip().split("\n")[0]

#check 2 inference server is reachable
@check("Inference server reachable")
def check_server():
    import urllib.request
    health_url = INFERENCE_ENDPOINT.rsplit("/v1", 1)[0] + "/health"
    try:
        with urllib.request.urlopen(health_url, timeout=5) as r:
            assert r.status == 200, f"health check returned {r.status}"
    except urllib.error.URLError as e:
        raise AssertionError(f"cannot reach {health_url}: {e}")
    return health_url

#check 3 nemotron responds and emits <think> tokens
THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)

def call_nemotron(prompt, max_tokens=512):
    """Minimal OpenAI-compatible chat completion call."""
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
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"]["content"]

@check("Nemotron returns output")
def check_nemotron_basic():
    text = call_nemotron("Say hello in five words.")
    assert text and len(text) > 0, "empty response"
    return f"got {len(text)} chars"

@check("Nemotron emits <think> reasoning trace")
def check_think_tokens():
    #a prompt that would reliably produce reasoning
    #adjust if reasoning parser config strips <think> tokens
    #in that situation we need to fix our server flags
    #for example: --reasoning_parser nano-v3
    text = call_nemotron(
        "A train leaves Chicago at 3pm going 60mph. Another leaves "
        "St. Louis at 4pm going 50mph. When do they meet? Show your reasoning."
    )
    match = THINK_PATTERN.search(text)
    assert match, (
        f"no <think>...</think> block found. Check your reasoning-parser "
        f"config. Response started with: {text[:200]!r}"
    )
    trace = match.group(1).strip()
    assert len(trace) > 20, f"think block too short: {trace!r}"
    return f"think block: {len(trace)} chars"

#check 4 is single step latency (informational for US, will not fail)
@check("Measure single-step latency")
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
    #not asserting a threshold
    #we are recording
    # need this number for the pitch and for
    #budgeting detector overhead
    return f"avg {avg:.2fs}s per step over {len(times)} calls"

#check 5 is openshell audit log exists and is writable
@check("OpenShell audit log accessible")
def check_audit_log():
    assert OPENSHELL_AUDIT_LOG.exists(), (
        f"audit log not found at {OPENSHELL_AUDIT_LOG}. Check NemoClaw install "
        f"or update the OPENSHELL_AUDIT_LOG constant in this script."
    )
    size = OPENSHELL_AUDIT_LOG.stat().st_size
    return f"{OPENSHELL_AUDIT_LOG} ({size} bytes)"

#check 6 is embedding model loads for goal drift detector
@check("Embedding model loads locally")
def check_embeddings():
    # sentence-transformers is easiest path
    # all-MiniLM-L6-v2 is 80MB
    # and fast enough to not bottleneck the agent loop
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise AssertionError(
            "sentence-transformers not installed. Run: "
            "pip install sentence-transformers"
        )
    model = SentenceTransformer("all-MiniLM-L6-v2")
    v1 = model.encode("the cat sat on the mat")
    v2 = model.encode("a feline was on the rug")
    v3 = model.encode("quantum mechanics is weird")
    # Sanity check: similar sentences should be more similar than unrelated ones
    import numpy as np
    sim_close = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    sim_far = np.dot(v1, v3) / (np.linalg.norm(v1) * np.linalg.norm(v3))
    assert sim_close > sim_far, (
        f"embedding sanity check failed: close={sim_close:.3f} far={sim_far:.3f}"
    )
    return f"close={sim_close:.3f} far={sim_far:.3f}"

#check 7 is one full agent step end-to-end
#this is the integration moment
#we do not require a real tool
#we just require that the model can produce a structured
#"I will call tool X with args Y" output that we will later parse
#if our openclaw agent is already wired up we need
#to replace this with an actual agent.run() call
@check("Full reasoning + tool-intent extraction")
def check_full_step():
    prompt = (
        "You are an agent with these tools: web_search(query), notes_write(text).\n"
        "Goal: find out who won the 2024 Nobel Prize in Physics.\n"
        "Think through your next step, then state EXACTLY which tool you'll call "
        "and with what arguments, in the form: TOOL: <name>(<args>)"
    )
    text = call_nemotron(prompt, max_tokens=512)
    think_match = THINK_PATTERN.search(text)
    assert think_match, "no reasoning trace in agent-style call"
    #look for tool-call line, this is what the reasoning-action
    #mismatch detector will parse
    tool_match = re.search(r"TOOL:\s*(\w+)\s*\((.*?)\)", text)
    assert tool_match, (
        f"no TOOL: line found in response. Response tail: ...{text[-300:]!r}"
    )
    tool_name = tool_match.group(1)
    return f"reasoning + tool call ({tool_name}) parsed cleanly"

#summary below
if __name__ == "__main__":
    print("=" * 60)
    print(" CoT Watchdog Setup Checkpoint")
    print(f" Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    #run checks in order
    #order matters
    #gpu -> server -> model -> embeddings -> integration
    #if gpu fails nothing else works
    check_gpu()
    check_server()
    check_nemotron_basic()
    check_think_tokens()
    check_latency()
    check_audit_log()
    check_embeddings()
    check_full_step()
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
        print("\n  RED. Fix failures before 8 PM checkpoint.")
        print("  Failures:")
        for name, ok, _, detail in results:
            if not ok:
                print(f"    - {name}: {detail}")
        sys.exit(1)
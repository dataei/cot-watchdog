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
import pathlib import Path

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

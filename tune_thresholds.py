import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from memory import list_all_tasks, load_task_timeline
from detectors.goal_drift import detect_goal_drift, _MODEL
from detectors.hedge import detect_hedge_miscalibration, HEDGE_PATTERN, CONFIDENCE_PATTERN
from detectors.mismatch import detect_mismatch
import numpy as np
def latest_task():
    tasks = list_all_tasks()
    if not tasks:
        print("No tasks in DB. Run agent_loop.py first.")
        sys.exit(1)
    return tasks[0]["task_id"]
def tune_goal_drift(steps, goal):
    """Sweep thresholds for the goal-drift detector and print results."""
    print("\n" + "=" * 60)
    print("  GOAL DRIFT THRESHOLD TUNING")
    print("=" * 60)
    goal_vec = _MODEL.encode(goal, normalize_embeddings=True)
    similarities = []
    for s in steps:
        step_vec = _MODEL.encode(s["reasoning"], normalize_embeddings=True)
        sim = float(np.dot(goal_vec, step_vec))
        similarities.append(sim)
        print(f"  Step {s['step_index'] + 1}: similarity {sim:.3f}")
    print("\n  Threshold sweep (which steps flag at each threshold):")
    for threshold in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        flagged = [i + 1 for i, sim in enumerate(similarities) if sim < threshold]
        print(f"    threshold={threshold:.2f}  →  flags steps: {flagged}")
    print("\n  Recommendation: pick a threshold that flags ONLY the steps")
    print("  matching your demo's planted goal-drift triggers.")
def tune_hedge(steps):
    """Sweep hedge density thresholds."""
    print("\n" + "=" * 60)
    print("  HEDGE MISCALIBRATION TUNING")
    print("=" * 60)
    for s in steps:
        trace = s["reasoning"]
        words = trace.split()
        if len(words) < 5:
            continue
        hedges = HEDGE_PATTERN.findall(trace)
        density = len(hedges) / len(words)
        conclusion = trace[int(len(trace) * 0.7):]
        conf = CONFIDENCE_PATTERN.findall(conclusion)
        print(f"  Step {s['step_index'] + 1}: hedges={len(hedges)} density={density:.3f} confidence_in_conclusion={len(conf)}")
    print("\n  Recommendation: HEDGE_DENSITY_THRESHOLD should be below the")
    print("  density of your planted miscalibration step but above clean steps.")
def show_mismatch_analysis(steps):
    """Show what the mismatch detector would see for each step."""
    print("\n" + "=" * 60)
    print("  REASONING-ACTION MISMATCH ANALYSIS")
    print("=" * 60)
    from detectors.mismatch import extract_stated_intent
    for s in steps:
        intent = extract_stated_intent(s["reasoning"])
        actual = s["tool_name"]
        match = "✓" if (intent and intent["tool"] == actual) else "✗"
        intent_str = intent["tool"] if intent else "NO INTENT FOUND"
        print(f"  Step {s['step_index'] + 1}: stated={intent_str}  actual={actual}  {match}")
    print("\n  The mismatch detector should fire on rows marked ✗.")
    print("  If it's missing some, check VERB_TO_TOOL coverage in intent_mapping.py.")
def main():
    task_id = latest_task()
    print(f"\nAnalyzing task: {task_id}")
    timeline = load_task_timeline(task_id)
    if not timeline:
        print("Task not found.")
        return
    goal = timeline["task"]["goal"]
    steps = timeline["steps"]
    print(f"Goal: {goal}")
    print(f"Steps: {len(steps)}")
    tune_goal_drift(steps, goal)
    tune_hedge(steps)
    show_mismatch_analysis(steps)
if __name__ == "__main__":
    main()
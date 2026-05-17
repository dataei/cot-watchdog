# CoT-Watchdog

Real-time chain-of-thought oversight for autonomous AI agents.

CoT Watchdog monitors an agent’s reasoning process live, detecting subtle failures before unsafe or unintended actions occur. The system runs fully on-device and introduces a human-in-the-loop checkpoint whenever suspicious reasoning behavior is detected.

Built during the NemoClaw NVIDIA x ASUS Hackathon at UCSC, on the ASUS Ascent GX10 using NVIDIA Nemotron Nano 3 and OpenClaw.

# Features
Goal Drift Detection

Tracks whether the agent’s reasoning is still aligned with the original task objective using embedding similarity and cosine distance.

Example:
Goal: “Summarize this paper”
Drifted reasoning: “Here’s my opinion on the topic”

If reasoning diverges too far from the original goal, the system flags the step before execution continues.

# Reasoning-Action Mismatch Detection
Compares what the agent says it is about to do against the actual tool call it makes.

Example:
Reasoning: “I’ll search for supporting evidence”
Actual action: calls unrelated tool

This catches silent inconsistencies between reasoning and execution.

# Confidence Miscalibration
Tracks hedge density and uncertainty language throughout the reasoning trace.

If the chain-of-thought contains significant uncertainty while the final answer is overly confident, the monitor flags the discrepancy for human review.

# Confidence Miscalibration Detection
Tracks hedge density and uncertainty language throughout the reasoning trace.

If the chain-of-thought contains significant uncertainty while the final answer is overly confident, the monitor flags the discrepancy for human review.

# Human-in-the-Loop Oversight
Whenever a monitor triggers:

 - execution pauses,
 - the full reasoning trace is surfaced,
- the operator must explicitly approve continuation.

All reasoning traces, alerts, and overrides are persisted for auditing.

# Tech Stack
- Python
- OpenClaw
- NVIDIA Nemotron Nano 3
- ASUS Ascent GX10 (DGX Spark)
- Sentence Transformers
- Vector Embeddings
- Persistent Trace Storage

# Why On-Device Matters
Reasoning traces are one of the most sensitive internal states of an AI system.

CoT Watchdog keeps all of the following fully local to the GX10:

- reasoning
- monitoring
- embeddings
- auditing

Nothing leaves the machine.

# Inspiration
This project was inspired by:

- “Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety” (Korbak et al., 2025)
- “CoT Red-Handed” (2025)
- Ataei et al., “Enhancing Autonomous Vehicle Test Scenario Reasoning in Language Models” (IEEE ITSC 2025)

The goal was to move chain-of-thought oversight from theory into a working real-time system.

# Challenges
Threshold Tuning: Finding cosine similarity thresholds that catch genuine drift without overwhelming operators with false positives.

Parsing Intent: Extracting actionable intent from free-form reasoning text is difficult because agents rarely state actions explicitly.

Latency: Running multiple monitors plus embeddings on every reasoning step while maintaining interactive responsiveness on-device.

Confidence Scoring: Designing interpretable uncertainty scoring without direct token probability access.

# Future Work
- Learned per-task drift thresholds
- Adversarial robustness testing
- Retrieval grounding verification
- Tool-sequence anomaly detection
- Multi-agent oversight
- Open-source monitoring framework

# Vision
Modern AI systems increasingly operate autonomously across long reasoning chains and tool-use workflows.

CoT Watchdog explores a simple question:

If agents reason in human language, can we monitor that reasoning before we trust the action?

This project is an attempt to build that oversight layer in practice.
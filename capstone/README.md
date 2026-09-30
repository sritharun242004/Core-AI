# Week 24 — build your capstone

Choose one track aligned with the two specialization tracks you studied. This repository supplies rubrics and deliverable templates, **not a pre-completed capstone**. Plan roughly 40 hours; at a 20-hour weekly pace, reserve two calendar weeks.

- [Research engineering rubric](track-research/RUBRIC.md)
- [Applied ML rubric](track-applied-ml/RUBRIC.md)
- [LLM product / agentic rubric](track-llm-product/RUBRIC.md)

## Seven work sessions

1. Scope: user/research question, success metric, baseline, non-goals, data license and compute ceiling.
2–3. Architecture and skeleton: make a minimal repo, write failing acceptance tests, establish baseline artifacts.
4–5. Core implementation: preserve raw metrics and failures; change one hypothesis at a time.
6. Evaluation: holdouts, ablations, failure slices, red-team where appropriate, cost and latency.
7. Writeup/demo: reproduce from a clean environment, self-score evidence and record remaining limits.

Create your own directory under `capstone/track-<name>/<your-project>/` with README.md, WRITEUP.md, DEMO.md, COMPUTE.md, SELF_RUBRIC.md, src/, tests/, evals/, configs/ and notebooks/. The empty application is yours to implement; no generated success claims belong in its report.

## Required artifacts

README: problem, setup, one-command reproduction, prerequisites and known limits.
WRITEUP: question, evidence, method, baseline, experimental protocol, results, failure analysis, limitations and extension.
DEMO: 3–5 minute storyboard, ordinary path, error path, measured result and evidence link.
COMPUTE: local/cloud requirements, current quote, hard stop conditions and teardown audit.
SELF_RUBRIC: for each criterion link an actual file, test output or experiment artifact, assign points, explain deductions.

Never upload credentials, private datasets or sensitive prompts. A public release or paid cloud execution remains an explicit learner decision. Scores assess evidence, not company interview guarantees.

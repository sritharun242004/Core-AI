# Timed interview rehearsals

These are self-study exercises, not guaranteed current questions or formats at named employers. Verify your actual recruiter instructions. AI assistance is allowed only where stated; disclose it and retain your review decisions.

## AI-assisted coding A — 45 minutes
Extend the Week 20 evaluator with grouped metrics and missing-label handling. Spend 5 minutes defining acceptance, 25 implementing with an assistant, 10 adversarial tests, 5 explaining rejected suggestions. Rubric (20): spec 4, correctness 6, tests 4, review 4, communication 2. Never paste private data or credentials.

## AI-assisted coding B — 45 minutes
Add bounded retry/idempotency to the Track L mock tool dispatcher. Ask the assistant for alternatives, then choose and test one. Rubric (20): side-effect model 5, bounded behavior 5, tests 5, explanation 5. Repeating a write after a timeout without an idempotency contract is a failure.

## AI-assisted coding C — 45 minutes
Investigate a ranking regression in Track P with an assistant. Require evidence for each proposed fix, preserve holdouts and show a rejected hypothesis. Rubric (20): isolation 5, leakage discipline 5, regression test 5, explanation 5.

## Take-home A — 90 minutes
Build an offline document-answer evaluator with exact-match baseline, case-level artifacts and tests. Timebox 10 requirements, 50 implementation, 20 tests, 10 README. Inspired by publicly discussed timed formats, not an assertion of an employer's current test. Deliver a clean reproduction command and known-limit list. Rubric (100): correctness 30, tests 25, simplicity 20, documentation 15, time management 10.

## Take-home B — four hours
Build a small retrieval UI around local documents with citations, failure states and an evaluation report. Timebox 30 scope, 120 implementation, 60 tests/evals, 30 demo/writeup. No external deployment or cloud spending required. Rubric (100): product 20, retrieval/evals 25, reliability 20, code 20, writeup 15. Preserve errors rather than hiding them behind an always-successful mock.

## Debugging drill A — 30 minutes
Introduce an unshifted next-token target in a disposable copy of Week 13/15a. Diagnose suspiciously good training and broken generation. Required evidence: target-alignment assertion, causal invariance test and before/after objective trace. Do not edit the canonical reference to leave it broken. Score: diagnosis 5, minimal fix 5, regression 5, explanation 5.

## Debugging drill B — 30 minutes
Introduce a future-derived feature in a copy of the applied pipeline. Identify why random CV looks excellent and chronological evaluation collapses. Required evidence: point-in-time join contract and leakage regression. Same 20-point rubric.

## Debugging drill C — 30 minutes
Average unequal local batch means in a copy of Week 18. Explain why equal-size tests pass and final-batch gradients differ. Required evidence: hand-computed weighted example and single-process gradient oracle. Same 20-point rubric.

## Review record

For each rehearsal record date, time limit, assistance policy, artifact commit, score by criterion, one misconception, a corrective drill and review date. Do not score confidence or eloquence instead of correctness. Repeat the weakest surface before adding harder questions.

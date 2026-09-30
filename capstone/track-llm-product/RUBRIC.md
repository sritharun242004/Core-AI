# LLM product / agentic capstone — 100 points

Choose a citation-backed research agent, bounded codebase reviewer, domain fine-tune+RAG hybrid, or voice-native pipeline. Reuse Track L protocols and Week 20/21 eval/safety contracts. A synthetic task harness is not a SWE-Bench Verified result.

| Criterion | 0–7: incomplete | 8–14: competent | 15–20: strong evidence |
| Product quality | Happy-path demo only | Usable UI and documented errors | Keyboard-accessible flow, loading/empty/error states, cancellation and visible evidence boundaries |
| Evaluation rigor | Hand-picked examples | Held-out task suite and regressions | Task outcomes, calibrated judge, red-team/overrefusal, failure slices and artifact provenance |
| Cost and latency | Unbounded calls | Token/call/time accounting | TTFT/total-latency distributions, retry/cache economics, measured budget ceilings and quality trade-offs |
| Agent design | Arbitrary tool execution | Typed allowlisted tools and step budget | Least privilege, untrusted-content separation, approval for side effects, idempotency and bounded recovery |
| Writeup and demo | Unsupported claims | Reproducible setup and demo | Honest comparison, protocol versions, operational runbook, privacy/retention and limitations |

## Evidence checklist

- Retrieval citations actually support answers, not merely plausible-looking URLs.
- Tool input/output schema validation; no shell/network execution controlled by arbitrary model text.
- Parallel calls only where dependencies and side effects permit them.
- MCP origin Anthropic; A2A/ADK Google Cloud. Simulations are not claimed as compliant deployments.
- Voice timing reports distinguish ASR, generation, TTS and transport; no unmeasured sub-second claim.

**Release gates:** no exposed secrets, uncontrolled side effects, fabricated evaluation results or unbounded spend. A learning target is 80/100 with each criterion at least 10.

**Demo:** one successful task, one injected/untrusted instruction rejected, one timeout or budget exhaustion, and the evidence/cost trace. **Stretch:** show a quality-preserving routing/cache change with a paired evaluation and real local timing.

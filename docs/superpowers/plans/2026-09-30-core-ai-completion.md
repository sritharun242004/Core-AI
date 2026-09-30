# Plans 7–8 — complete the curriculum

## Delivery contract

Plan 7 adds Weeks 18–21 and tags `v0.7.0-week21`; Plan 8 adds six specialization lessons/projects, Week 24 capstone rubrics and Week 25 interview protocol, then tags `v1.0.0`. Both fast-forward into `plan-1-foundation`. Do not deploy or run cloud workloads.

### Plan 7

- W18 `week-18-fsdp-ring-lab`: state/communication ledgers, DP equivalence, exact ring-order attention, optional FSDP/ZeRO entrypoints.
- W19 `week-19-vllm-benchmark`: quantization, cache accounting, decoding and optional measured backend adapters.
- W20 `week-20-evals-mlops-pipeline`: versioned data/evaluators, judge calibration, paired uncertainty, drift and regression gates.
- W21 `week-21-alignment-lab`: DPO, twenty harmless synthetic red-team fixtures, logit lens, probe and tiny SAE.

### Plan 8

Six distinct track lessons use numeric weeks 22/23 plus track metadata L/P/R. Navigation must preserve all alternatives rather than silently overwriting equal week numbers.

| Track | Alpha (week 22) | Beta (week 23) |
|---|---|---|
| L: LLM product | `week-22l-agents-lab` | `week-23l-mcp-a2a-adk-lab` |
| P: Applied ML | `week-22p-two-tower-recsys` | `week-23p-learning-to-rank-timeseries` |
| R: Research | `week-22r-transformer-repro` | `week-23r-scaling-dpo-repro` |

Each slug corresponds to a Python project of the same name and one MDX route. Track R's two projects together cover Transformer, mini scaling-law studies and Stanford DPO; synthetic offline fixtures are not full paper reproductions or measured Chinchilla exponents. Optional larger-scale data/compute instructions must be explicitly bounded.

- `week-24-capstone`: three complete rubric templates under `capstone/track-{research,applied-ml,llm-product}`; learners build their own capstones, not a fake pre-completed project.
- `week-25-interview-prep`: seven-day protocol with timed coding, fundamentals, system design, company evidence, behavioral/values, mock loops, AI-assisted work and outreach templates. Do not send outreach automatically.

## Scope reconciliation

The historical outline calls this a 25-week roadmap, but choosing two two-week tracks takes four specialization weeks and the capstone is approximately 40 hours. Keep roadmap labels without pretending the schedule is literally 25 calendar weeks at 20 hours/week. Explain the extra specialization/capstone time in the study guide. Shipping every choice produces 29 lesson routes and 27 reference packages; W16 stays absent.

## Acceptance

Test numerical contracts before implementing; use deterministic offline fixtures; assert seven-company parity and paper/tool attribution; execute every new notebook; validate schemas/navigation/track links/project paths, KaTeX and Pagefind; fix or explicitly document outstanding platform issues. Use Node 24 and Python 3.13. Confirm full per-project pytest, JS unit, browser, build and lint gates before release. Git identity remains sritharun242004 with no co-author trailers. Keep harness metadata untracked.

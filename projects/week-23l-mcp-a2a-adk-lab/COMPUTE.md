# Compute and spend boundaries

**Curriculum tier: yellow. Required reference: CPU-only, $0.** Python 3.13 and
pytest/Ruff are sufficient. No runtime dependencies, downloads, API calls, model
weights or GPU are required. Tests/notebook should finish in seconds with tiny
in-memory fixtures; this is an estimate, not a recorded hardware benchmark.

## Optional ADK path — not executed in the reference verification

1. Create a separate disposable environment and install this project's optional
   `[adk]` extra, pinned to `google-adk==1.18.0`. Do not mutate the shared repo venv.
2. Follow the pinned Google Cloud ADK documentation for Gemini API or Vertex AI
   authentication. Use a dedicated low-privilege project; never commit credentials.
3. Check available model IDs and a dated provider rate card. The sample's default
   string is illustrative, not a guarantee of current model availability.
4. Set a small project quota/spend limit, e.g. **$1 total for the experiment**, and
   calculate a per-run upper bound from input/output limits and at most two model
   calls. Budget alerts alone may not enforce a hard stop; use enforceable quotas.
5. Construct the agent first without `--run-model`. Only then opt into model calls.
   The runner has a two-call cap and a best-effort 30-second client deadline.
   A canceled client request may still incur provider charges.
6. Inspect/redact logs; revoke temporary credentials, stop any local process and
   delete any experiment resources. This sample creates no cloud deployment.

No production sandbox, server authentication or remote agent transport is supplied.
The $/million constants in economics examples are **hypothetical**, not a quote
for Google, Anthropic, OpenAI or another provider. A real experiment must validate
cache eligibility, all-in versus surcharge write pricing, batch discount rules,
tokenizer accounting and retry billing for its chosen provider and date.

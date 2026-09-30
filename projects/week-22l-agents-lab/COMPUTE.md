# Compute — yellow curriculum, green reference execution

- Required: Python 3.13, a CPU, pytest for tests; zero runtime dependencies.
- Reference suite/notebook: typically under a few seconds, well below 100 MB for
  these small fixtures. Treat this as a scale estimate, not a hardware benchmark.
- Required spend: **$0**. No model downloads, API calls, credentials or cloud runs.
- Yellow reflects optional real-model experiments, not a hidden required GPU.

If extending with a hosted model, create a separate opt-in adapter/environment.
Pin model/provider versions, enforce maximum input/output tokens, tool/step caps,
a wall-clock deadline and a prepaid experiment budget (for example $2 total).
Calculate the maximum possible spend before enabling any call; stop rather than
silently falling back to a more expensive model. Never place secrets in notebooks.
No cloud run was performed or validated for this reference. Threaded callbacks
need separate process isolation to enforce hard timeouts; a call count is not a
latency bound. Run only trusted local fixtures in the shipped implementation.

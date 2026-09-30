# Compute — yellow curriculum tier, local core

- Python 3.13, NumPy and PyTorch. CPU-only correctness tests and percent notebook; zero credentials/downloads/API calls.
- Ranking has 180 synthetic query-document rows. Pair construction is O(documents²) within a query, intentionally not a production training strategy.
- Forecasting trains small two-block networks on fewer than 100 time points; it refits at each origin so the information boundary is easy to audit. The notebook uses one Torch CPU thread and a small number of origins.
- Expect seconds to tens of seconds per full run on a typical laptop; these are planning estimates, not measured production benchmarks. Memory is comfortably below 1 GB beyond framework overhead.
- Cloud budget: **$0**. No provisioning and no teardown required. Real rerankers, large panels and hyperparameter sweeps are optional learner work, not required cloud runs.
- No LightGBM/LambdaMART, TFT or Prophet dependency is hidden behind an extra. Adding those packages requires explicit local installation, pinned versions, group/time-safe adapters and separate evaluation. No benchmark accuracy or speed is promised.
- For voluntary cloud experiments, set a spending/time limit, save only intended artifacts, and terminate instances plus attached storage. Do not confuse forecast uncertainty with cloud-run variability.

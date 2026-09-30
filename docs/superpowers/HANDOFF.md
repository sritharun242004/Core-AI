# Core AI — Development Handoff

**For:** any AI operator (Claude, GPT, or similar) resuming development on this repo.
**From:** the operator that completed all eight curriculum plans through `v1.0.0`.
**Date:** 2026-09-30.
**Repo:** https://github.com/sritharun242004/Core-AI (private) · main working branch: `plan-1-foundation`.

Read this file top-to-bottom before touching anything. Skimming will bite you.

---

## 1. What this project is

An open-source, self-driven 25-week AI curriculum for full-stack engineers targeting frontier-lab roles.

The deliverable is a book (`apps/book/`) with **29 lesson routes, 27 Python reference projects, seven company profiles, three capstone rubrics and interview workbooks**. The 25-week roadmap has six alternative specialization lessons; learners choose two tracks. W15a/15b replace W16. Four specialization weeks and a 40-hour capstone mean roughly 28+ calendar weeks at 20 hours/week. There is no backend/auth/per-user server state; `/companies/compare` remains SSR.

**Design spec:** `docs/superpowers/specs/2026-09-22-core-ai-book-design.md` — this is the binding authority. Every decision in the codebase should trace back to a spec section. If the spec is ambiguous, prefer the interpretation that ships something readable, then flag the ambiguity in your handoff note.

**Plans done:**
- Plan 1 (`docs/superpowers/plans/2026-09-22-core-ai-foundation.md`) — foundation + Week 1 pipeline proof. Shipped, tagged `v0.1.0-week01`.
- Plan 2 (`docs/superpowers/plans/2026-09-27-core-ai-platform-features.md`) — companies subsystem, interview dashboard, Motion, viz package, Pagefind, companion routes. Shipped, tagged `v0.2.0-platform`.
- Plan 3 (`docs/superpowers/plans/2026-09-28-core-ai-weeks-2-4.md`) — Weeks 2-4 (calculus, probability, python+info-theory) + 3 Python reference projects. Shipped, tagged `v0.3.0-week04`.
- Plan 4 (§5 below) — Weeks 5-8 (classical ML) + 4 Python reference projects. Shipped, tagged `v0.4.0-week08`.
- Plan 5 (§6 below) — Weeks 9-12 (deep learning foundations) + 4 Python reference projects. Shipped, tagged `v0.5.0-week12`.
- Plan 6 (§7 below) — Weeks 13-17 (transformers, LLMs, post-training) + 5 Python reference projects. Shipped, tagged `v0.6.0-week17`.

- Plan 7 (§8 below) — Weeks 18–21 (systems + safety), four reference projects. Shipped as `v0.7.0-week21`.

- Plan 8 — all six specialization lessons/projects, capstone rubrics and interview prep, shipped as `v1.0.0`. See §9 and `plans/2026-09-30-core-ai-completion.md`.

**Plans remaining: none.** Optional cloud experiments, learner-built capstones and deployment are not executed on the learner's behalf. Strict Python typing remains technical debt, explicitly separated from curriculum completion and runtime validation.

---

## 2. Current state (complete curriculum, tag `v1.0.0`)

### 2.1 Test snapshot
- Viz Vitest: **3/3**
- Book Vitest: **22/22**
- Playwright: **102/102**, no skips; former motion fixmes are now real passing tests.
- Python pytest: **676/676 across all 27 packages**, zero failures/errors/skips, using `scripts/check-projects.py`.
- All **27 percent-format notebooks** executed offline, including a new Week 1 synthetic SVD companion.
- Repository Biome and Python Ruff checks pass. Astro/Pagefind builds **46 indexed static pages**.
- Link checker passes **60 unique external URLs**; all emitted local links validated. The verified OpenAI SWE-bench article is explicitly exempted from bot-blocked HEAD requests after a successful readable GET.

**Total: 803 passing automated tests.** Dependency synchronization succeeded with `uv sync --all-packages --extra dev`; numerical/data/model tests require no network. Cold dependency/font builds and external link validation do need network. **Strict Pyright is not clean** and remains a visible advisory CI audit; see `docs/VALIDATION.md`.

### 2.2 What ships in the built book
- 46 prerendered static pages (`dist/client/`) + `/companies/compare` and the `/motion-probe` test vehicle as server routes
- Pagefind client-side search (⌘K), index freshness pinned by test
- 29 lessons, including all L/P/R α/β options and W24/W25, each with lesson anatomy and seven-company CompanyLens; no W16
- 7 company profiles (`/companies/[slug]`), all attribution-clean per spec §5.3
- 7×25 CompanyLens matrix: all 175 comparison cells populated, all six track links preserved in W22/W23 columns
- `/interview` dashboard + `/interview/coding-set` (20 problems) + weak-spot heatmap
- 5 companion routes: `/how-to-study`, `/glossary`, `/math-primer`, `/paper-reading-protocol`, `/tech-writing`
- Interactive islands: MicroRecall, WeeklyQuiz, InterviewDashboard, WeakSpotHeatmap, ThemeToggle, SearchDialog — all Preact + Signals

### 2.3 What's local-only (not yet deployed)
- Not deployed to Vercel. The user will handle this under their own Vercel account. Do not run `vercel --prod` unless explicitly asked.
- GitHub remote exists (`origin` → https://github.com/sritharun242004/Core-AI); `plan-1-foundation` and all eight milestone tags through `v1.0.0` are pushed after completed plans.

---

## 3. Ground rules for AI operators

Read these once; they are the difference between shipping and thrashing.

1. **The spec is binding.** Never invent facts about companies, papers, or models beyond `docs/superpowers/specs/2026-09-22-core-ai-book-design.md` §5.3. When in doubt, leave the reference blank with `<!-- TODO: verify -->` — never guess.
2. **Attribution rules are verbatim.** See §11. Violating one is a Critical bug caught by `tests/company-attribution.spec.ts`.
3. **7-company parity is a build-time contract.** Every `<CompanyLens>` block must supply all 7 slugs (openai, anthropic, deepmind, meta, xai, deepseek, qwen) in `tldr`, `sources`, `interviewAngle`. `assertAll7` in `apps/book/src/lib/companies.ts` fails the build if you miss one.
4. **Uniform Python project shape.** See §10.3. Every new project must follow it — no exceptions.
5. **Test-driven for correctness code**, but MDX and prose don't need failing tests. Follow §10.4.
6. **Never push to `main`** — no `main` branch exists. Work on `plan-1-foundation` or a feature branch and merge back with `--ff-only`.
7. **Never `vercel --prod`** without explicit user consent.
8. **Never `--force`** on any destructive git command without explicit user consent.
9. **Commit at every task boundary.** Small, reviewable commits. Follow the `feat(wXX):` / `feat(book):` / `test(...)` prefix convention from git log.
10. **When you deviate from a plan, ledger the ruling** in a file named `.superpowers/sdd/<plan-basename>/progress.md` — this directory is git-ignored.

---

## 4. Repo tour

```
Core-AI/
├── README.md                    (project intro, links to spec/plans)
├── package.json                 (pnpm workspace root)
├── pnpm-workspace.yaml          (apps/*, packages/*)
├── pyproject.toml               (uv workspace root; members = ["projects/*"])
├── turbo.json, biome.json, ruff.toml
├── .node-version → 24           (Node 24 LTS required — session default is often 20; use nvm)
├── .python-version → 3.13
├── docs/superpowers/
│   ├── specs/2026-09-22-core-ai-book-design.md    (THE binding authority)
│   ├── plans/                   (per-plan implementation plans)
│   └── HANDOFF.md               (this file)
├── apps/book/                   (Astro 6 site)
│   ├── astro.config.mjs         (Vercel adapter, motion aliased react→preact/compat)
│   ├── vitest.config.ts         (Preact preset for .spec.tsx)
│   ├── playwright.config.ts     (webServer: astro dev, workers:1)
│   ├── package.json             (scripts: dev/build/postbuild/preview/test/test:unit)
│   ├── src/
│   │   ├── content.config.ts    (Zod schemas for weeks/companies/extras)
│   │   ├── content/
│   │   │   ├── weeks/*.mdx      (29 complete routes; W16 intentionally absent)
│   │   │   ├── companies/*.mdx  (7 profiles, all done)
│   │   │   └── extras/*.mdx     (5 companion pages, all done)
│   │   ├── lib/
│   │   │   ├── companies.ts     (COMPANIES const, CompanySlug type, assertAll7)
│   │   │   ├── progress.ts      (localStorage-safe progress store; use recordAnswer, getMastery, surfaceCounts, totalSolved)
│   │   │   └── quiz.ts          (SM-2 lite nextDueDate)
│   │   ├── components/
│   │   │   ├── content/         (8-part lesson anatomy Astro components)
│   │   │   ├── callouts/        (Intuition/Gotcha/DeepDive/Interview/CompanyPill)
│   │   │   ├── companies/       (CompanyMatrix, CompanyProfileHeader, LoopGuide, CompanyReadingList, CompareTable)
│   │   │   ├── interactive/     (MicroRecall, WeeklyQuiz, InterviewDashboard, WeakSpotHeatmap, SearchDialog, ThemeToggle) — Preact + Signals
│   │   │   ├── motion/          (ScrollReveal — reduced-motion via custom useReducedMotionSafe)
│   │   │   └── layout/          (BookShell, Sidebar, Footer, ReadingProgressBar)
│   │   ├── layouts/             (WeekLayout, CompanyLayout, ExtraLayout)
│   │   └── pages/
│   │       ├── index.astro      (home — hero + 25-week roadmap)
│   │       ├── weeks/[slug].astro
│   │       ├── companies/{index,[slug],compare}.astro
│   │       ├── interview/{index,coding-set}.astro
│   │       ├── {how-to-study,glossary,math-primer,paper-reading-protocol,tech-writing,search,motion-probe}.astro
│   └── tests/                   (Playwright *.spec.ts, vitest *.spec.tsx and *.unit.spec.ts)
├── packages/viz/                (@core-ai/viz workspace pkg — Preact SVG primitives)
│   ├── src/{index,shared,VectorPlayground,MatrixMul}.tsx
│   └── tests/viz.spec.tsx
├── projects/                    (27 complete Python packages; W16 intentionally absent)
│   ├── week-01-linalg-lab/
│   ├── week-02-micrograd/
│   ├── week-03-prob-lab/
│   └── week-04-numpy-vs-pytorch/
├── scripts/check-links.mjs      (used by weekly link-check CI cron)
└── .github/workflows/
    ├── book.yml                 (vitest + viz + astro build + playwright)
    ├── projects.yml             (per-project uv sync + pytest)
    ├── ci.yml                   (repo-wide biome + ruff)
    └── link-check.yml           (weekly cron)
```

---

## 5. Plan 4 — Weeks 5-8 (classical ML)

**Goal:** Ship 4 MDX articles + 4 Python reference projects for Month 2 (classical ML).

**Status:** Shipped and validated in `v0.4.0-week08`.

**Compute tier:** 🟢 for all 4 weeks (M-series, no cloud needed).

**Weeks and their reference projects:**

| Week | Topic | Project slug | Focus |
|---|---|---|---|
| W5 | Linear & logistic regression, gradient descent from scratch | `week-05-linreg-from-scratch/` | GD + logistic, no sklearn |
| W6 | Trees, ensembles, XGBoost, SVMs | `week-06-xgboost-kaggle/` | Full Kaggle submission pipeline |
| W7 | Unsupervised — k-means, PCA, GMM, t-SNE, UMAP | `week-07-unsupervised-viz/` | 4 methods on same dataset, side-by-side viz |
| W8 | Model evaluation, bias/variance, CV, feature engineering, leakage | `week-08-ml-eval-suite/` | CV, learning curves, calibration, feature importance |

### 5.1 Week 5 outline
- **MDX (`week-05-linreg-from-scratch.mdx`):** normal equation vs GD, MSE loss derivation, logistic loss + sigmoid, regularization (L1/L2, elastic net). CompanyLens: OpenAI (RLHF starts from a logistic classifier), Anthropic (RSP scoring uses regularized logistic), DeepMind (AlphaFold uses linear last layer for coords), Meta (feed ranking is logistic at scale), xAI (Grok's ad matching linear), DeepSeek (any linear head — routing gate), Qwen (linear projection heads in Qwen-VL).
- **Project:** implement `LinearRegression`, `LogisticRegression` with `.fit(X, y)`, `.predict(X)`, `.predict_proba(X)`, `.coef_`. Tests: (a) closed-form solution on synthetic linear data → coefficients within 1e-6; (b) GD converges to same solution in < 500 steps; (c) logistic loss decreases monotonically; (d) L2 regularization actually shrinks coefficients.
- **Assignments:** warmup (30 min) plot a learning curve; build (2-3 hr) sklearn-competitive logistic on Titanic dataset; challenge (3+ hr) manually derive + implement elastic-net proximal gradient.

### 5.2 Week 6 outline
- **MDX:** decision trees (impurity criteria: Gini, entropy), bagging → random forests, boosting → XGBoost (gradient boosting, additive stages, second-order Newton step), SVMs (max-margin, kernel trick, RBF vs linear). CompanyLens: everyone uses trees for tabular internal metrics; OpenAI's feature stores still gradient-boost; Anthropic uses tree-based classifiers for RSP evals; DeepMind's tabular experiments; Meta uses GBDT in ads ranking pre-DL layer; xAI unknown; DeepSeek n/a; Qwen n/a.
- **Project:** `week-06-xgboost-kaggle/` — a Titanic (or California housing) end-to-end pipeline: EDA notebook → feature engineering → sklearn baseline → XGBoost model → cross-validation → submission CSV. Tests: (a) trained model beats baseline by ≥ 5% ROC-AUC; (b) submission file has correct schema.
- **Reading:** *Hands-On ML with Scikit-Learn, Keras & TensorFlow (3E, 2022)* — Géron, chapters 6-7.

### 5.3 Week 7 outline
- **MDX:** k-means (Lloyd's algorithm, initialization: k-means++), PCA (SVD of centered data, explained variance ratio), GMM (EM algorithm, soft assignments), t-SNE (probability-preserving embedding), UMAP (topological embedding, faster than t-SNE). CompanyLens: OpenAI uses embeddings for retrieval; Anthropic Circuits uses SAEs (dictionary learning is unsupervised); DeepMind's AlphaFold uses PCA on MSAs; Meta uses unsupervised for cold-start recsys; DeepSeek's V3 latent attention is dimensionality reduction.
- **Project:** `week-07-unsupervised-viz/` — 4 methods (k-means, PCA, t-SNE, UMAP) on the same digits/faces dataset, side-by-side matplotlib figure. Tests: (a) PCA reconstruction preserves variance; (b) k-means with k=3 on iris finds species clusters (adjusted rand ≥ 0.7).
- **Compute:** UMAP requires `umap-learn` — mark it as optional; the notebook works without it (just no UMAP panel).

### 5.4 Week 8 outline
- **MDX:** train/val/test splits, cross-validation (k-fold, stratified, time-series), bias-variance decomposition, learning curves, feature engineering, target/data leakage (the top-cause of "too-good-to-be-true" models). CompanyLens: OpenAI holdout-set discipline; Anthropic's eval suite (Inspect AI); DeepMind's cross-val protocols in AlphaFold; Meta's A/B testing for ranking changes; xAI cluster benchmarks; DeepSeek's contamination detection; Qwen's benchmark leak audits.
- **Project:** `week-08-ml-eval-suite/` — a Callable suite that takes any `(estimator, X, y)` and returns: (a) 5-fold CV metrics; (b) learning curve dict; (c) calibration curve data; (d) permutation feature importance; (e) leakage detector (checks train/test target correlation). Tests: (a) CV metrics match sklearn's `cross_val_score`; (b) leakage detector flags a hand-crafted leak.
- **Deliverable importance:** this is the tool you use in every remaining week to validate your models. Ship it well.

### 5.5 Plan 4 execution recipe
1. Extend company `seededAngles` for weeks 5-8 (one entry per company per week where relevant — most companies get 1-2 entries in this range).
2. For each week in order:
   a. Scaffold `projects/week-0X-<name>/` per §10.3.
   b. Write pyproject, source module(s), and tests. Run tests via `uv run --extra dev pytest` per project (§10.6).
   c. Write README, SOLUTION_NOTES, COMPUTE, 3 assignment files.
   d. Write `notebooks/01-*.py` (percent-format).
   e. Write `apps/book/src/content/weeks/week-0X-<name>.mdx` per §10.1.
   f. Run `pnpm --filter book run build` to verify prerender + PagefindI index.
   g. Commit as `feat(w0X): <topic> MDX + project`.
3. Add `apps/book/tests/weeks-5-8-render.spec.ts`, extend `company-lens-parity.spec.ts` to include weeks 5-8, add `katex-weeks-5-8.spec.ts`.
4. Run all suites (§10.6). Tag `v0.4.0-week08`.
5. Merge back to `plan-1-foundation`, push, push tag.

**Plan 4 validation actually run:** `pnpm --filter book run build` (all W1-W8 routes + Pagefind), viz vitest 3/3, book vitest 6/6, Playwright 42/42 + 2 fixmes, per-project pytest 85/85, Week 5-8 Ruff clean, and `pnpm run check-links` clean. The historical shell-loop URL `http://localhost:4321/$s` is ignored by the link checker because it is a planning-doc artifact, not a clickable link.

**Estimated size:** 4 MDX × ~180 lines + 4 projects × ~500 lines = ~2900 lines new. Tag `v0.4.0-week08`.

---

## 6. Plan 5 — Weeks 9-12 (deep learning foundations)

**Status:** Shipped and validated in `v0.5.0-week12`.

**Compute tier:** 🟢 W9, W11, W12; 🟡 W10 (CIFAR ResNet — 90% locally, 93%+ needs cloud/MLX).

| Week | Topic | Project | Notes |
|---|---|---|---|
| W9 | Neural nets from scratch + backprop derivation | `week-09-mini-torch/` | NumPy micro-torch → MLP on an offline MNIST-shaped fixture |
| W10 | CNNs, ResNets, augmentation, transfer learning | `week-10-cifar-resnet/` | 🟡 ResNet-18 from scratch → 90% on MPS in ~1 hr; 93%+ needs cloud/MLX overnight |
| W11 | RNNs, LSTMs, seq2seq, attention mechanism | `week-11-char-rnn-attention/` | Char-level LSTM/GRU + additive attention on a deterministic tiny text fixture |
| W12 | Optimizers (SGD/Adam/AdamW), regularization, RL primer (MDPs, Q-learning, policy gradient, PPO) | `week-12-rl-gridworld/` | Q-learning + policy gradient on gridworld. VAE/GAN/diffusion is moved to W17, DO NOT put it here. |

**Key spec note (§5.1 note on W10):** honesty about compute. Set the expectation that 90% CIFAR is realistic locally; higher accuracy needs cloud. Add a COMPUTE.md line.

**Key spec note (§ Section 1 W12):** RL primer (MDPs, Q-learning, PG, PPO) IS in W12. Generative models (VAE/GAN/diffusion) are NOT — those move to W17. Do not accidentally include them.

**Attention (W11):** Karpathy-style additive attention on a char-level LSTM. This is the bridge to W13's transformer — the whole point of W11's attention section is to make W13 feel inevitable.

**Validation actually run:** Astro build + Pagefind (29 indexed HTML pages), viz vitest 3/3, book vitest 6/6, Playwright 54/54 + 2 fixmes, per-project pytest 144/144, Weeks 9-12 Ruff clean, all four notebooks executed offline, and `pnpm run check-links` clean. Tag `v0.5.0-week12`.

---

## 7. Plan 6 — Weeks 13-17 (transformers, LLMs, post-training)

**Status:** Shipped and validated in `v0.6.0-week17`.

**This is 5 weeks, not 4** — spec §1 explicitly split W15 into W15a + W15b to be honest about scope. Cloud-GPU weeks start here.

| Week | Topic | Project | Tier | Cost |
|---|---|---|---|---|
| W13 | Attention Is All You Need — transformer from scratch in PyTorch; **post-transformer landscape (SSMs / Mamba / Jamba / Griffin)** | `week-13-nano-gpt-ssm/` | 🟡 | ~$0-10 (1-2M param GPT on any M-series in 30 min; 10M variant on M3+ Max with MLX; side-by-side with a tiny Mamba/SSM baseline) |
| W14 | Tokenization, embeddings, **positional encoding depth (RoPE / ALiBi / YaRN, long-context tricks)**, pre-training vs post-training paradigms, weights & checkpoints | `week-14-mini-bpe-pretrain/` | 🟡 | ~$5-15 (BPE from scratch + TinyStories pretraining, 3-5 hrs single GPU) |
| W15a | **Post-training foundations** — SFT, RLHF, **DPO + KTO/IPO/ORPO/SimPO**, RLAIF, Constitutional AI, **LoRA / QLoRA / DoRA / PEFT**, synthetic-data pipelines | `week-15a-sft-lora-dpo-lab/` | 🔴 | ~$20-50 (Llama-3.2-1B local or Llama-3-8B cloud) |
| W15b | **Advanced post-training** — LLM families (GPT/Claude/Llama/Mistral/Qwen/DeepSeek/Gemma/Kimi/GLM), scaling laws, **MoE deep dive**, **reasoning models & test-time compute (o-series / R1)**, **GRPO / RLVR / verifiers / DualPipe** | `week-15b-moe-and-reasoning/` | 🔴 | ~$20-50 (tiny MoE from scratch + GRPO training loop on GSM8K subset) |
| W17 | Multimodal — CLIP, ViT, **VAE/GAN/diffusion (moved from W12) + world/video models**, audio + **voice-native pipelines**, **RAG deep dive + named variants (GraphRAG, HyDE, ColBERT)**, semantic search, re-ranking, retrieval metrics, knowledge-graphs intro | `week-17-mini-rag-multimodal/` | 🟡 | ~$5-15 (end-to-end RAG with evals + tiny VAE/diffusion + voice pipeline) |

**Critical spec notes:**
- W13 must cover BOTH transformers AND post-transformer alternatives (Mamba, Griffin) — the spec calls this out explicitly.
- W14 must go deep on positional encodings (RoPE, ALiBi, YaRN) — this is where the interview questions live.
- W15a's DPO must be attributed to **Stanford** (Rafailov et al. 2023). Not Meta. See §11.
- W15b covers GRPO — introduced in DeepSeekMath (arXiv:2402.03300) and later used in DeepSeek-R1 (arXiv:2501.12948). Cite both papers directly.
- No W16. The spec renumbered W16 → W17 when the split happened. Don't create a phantom week 16.

**Cloud runbooks:** each 🔴 project needs a `COMPUTE.md` that says:
- Provider (RunPod / Modal / vast.ai)
- Instance type (A100 40GB is typical baseline)
- Estimated hours + cost
- How to teardown (this is what saves users money)

**Validation actually run:** Astro build + Pagefind (34 indexed HTML pages), viz vitest 3/3, book vitest 8/8, Playwright 69/69 + 2 fixmes, per-project pytest 355/355, Plan 6 Ruff clean, all five notebooks executed offline, and `pnpm run check-links` clean. Tests use no network, model downloads, cloud GPU, or API credentials. Tag `v0.6.0-week17`.

---

## 8. Plan 7 — Weeks 18-21 (systems + safety)

**Status:** shipped as `v0.7.0-week21`.

W18/W19 cloud extensions are red-tier; all shipped correctness tests and notebooks run offline on CPU. W20/W21 are yellow-tier. Real multi-GPU training, HF/vLLM checkpoint inference and external tracking/provider integrations were not executed.

**Fresh validation:** 164 Python tests (70/51/16/27), four offline notebooks, new-project Ruff checks, 3 viz and 8 book unit tests, Astro/Pagefind with 38 pages, 78 browser tests plus 2 pre-existing fixmes. Link checking encountered one transient Google documentation timeout; the live URL was independently fetched successfully and the rerun passed all 33 URLs.

**Rulings:** D2 architecture sources plus explicitly hand-authored accessible SVG fallbacks (D2 CLI unavailable); Inspect AI correctly attributed to UK AI Security Institute; scientific/serving claims distinguish simulations from measured results. Completion plan explains the historical 25-week versus specialization/calendar-time mismatch.

| Week | Topic | Project | Tier |
|---|---|---|---|
| W18 | Distributed training (data / tensor / pipeline parallelism, FSDP, DeepSpeed, Megatron, **ring / context-parallel attention**), **GPU/TPU internals + CUDA / Triton kernel intro** | `week-18-fsdp-ring-lab/` | 🔴 |
| W19 | Inference engines (vLLM, TGI, TensorRT-LLM, Triton, llama.cpp, **plus Groq / Cerebras / SambaNova as alternative silicon**), quantization (INT8/INT4, GPTQ, AWQ, KV-cache quant), **KV cache paging**, speculative decoding, latency vs throughput (TTFT, TPOT, batching), **prompt & context caching** | `week-19-vllm-benchmark/` | 🔴 |
| W20 | **MLOps + Evaluation Infrastructure** — data pipelines, feature stores, monitoring, CI/CD for ML, synthetic-data pipelines; **LLM-as-judge, MT-Bench, HELM, MMLU-Pro, SWE-Bench, Inspect AI, Harbor, trajectory & agent evals, contamination detection** | `week-20-evals-mlops-pipeline/` | 🟡 |
| W21 | AI safety & alignment — RLHF deep, Constitutional AI, red-teaming batteries, **interpretability (logit lens, probing, sparse autoencoders / dictionary learning — Anthropic Circuits methods)**, safe-agent foundations | `week-21-alignment-lab/` | 🟡 |

**Critical spec notes:**
- W18 diagrams: use D2 (spec §2.2) for FSDP/ring-attention architecture. It's better than Mermaid for these.
- W19: silicon-alternatives section is a differentiator — Groq / Cerebras / SambaNova are named companies but NOT among the 7 first-class labs; treat them as supporting cast per spec §4.1.
- W20 evals: this is where W2-W15's work gets validated. Reference back to earlier weeks' models by name.
- W21 alignment: use DPO (not PPO) for the tiny RLHF loop. Use logit-lens + tiny SAE (not full Circuits) for the interp probe. Spec §5.1 explicitly says this.

**Estimated size:** 4 MDX × 250 lines + 4 projects (W18-19 are the hardest — real cloud engineering). Tag `v0.7.0-week21`.

---

## 9. Plan 8 — Weeks 22-25 (specialization + capstone + interview prep)

**Status: complete in `v1.0.0`.** Six full track lessons/packages ship: L α/β (33/46 Python tests), P α/β (18/27), R α/β (11/22 after independent-review regressions). W24 supplies three complete 100-point rubrics, and W25 supplies the seven-day protocol plus eight timed rehearsals. No fallback/pending track is used.

Research fixtures explicitly distinguish mechanism parity from full WMT/TinyStories/Chinchilla replication. MCP is a documented 2024-11-05 teaching subset; the A2A lifecycle is explicitly a noncompliant local subset; optional real ADK wiring is fake-tested only. No model/data downloads or cloud experiment was executed.

Final platform work repairs track navigation, persistent quiz schedules/IDs, malformed progress records, reduced-motion/no-JS rendering, local Pagefind search, repo links, prerequisites and mobile overflow. CI now runs pytest in isolated project processes and excludes generated caches from source linting. The inherited strict-typing backlog is documented rather than misrepresented as passing.

**Weeks 22-23 = 4 weeks total, learner picks 2 of 3 tracks:**

Track L — LLM Product / Agentic (2 weeks):
- Week α · Agents & agentic AI — architectures (ReAct/Reflexion/Plan-and-Execute), tool use, planning, memory, long-horizon tasks, multi-agent, function calling / structured output / tool schemas / parallel tools
- Week β · Protocols & frameworks — **MCP, A2A, ADK (Google Cloud products, not DeepMind — spec §5.3)**, LangGraph, CrewAI, AutoGen, OpenAI Agents SDK, Knowledge-Graph RAG, prompt & context engineering, **agent evals (SWE-Bench Verified, Inspect AI, trajectory metrics)**, **cost engineering / LLM economics**

Track P — Applied ML (2 weeks):
- Week α · Recsys, ranking, ads ML, search ranking, retrieval at scale, GNNs (recsys, fraud, molecules, social)
- Week β · Learning-to-rank + neural rerankers + A/B testing + causal inference basics, **time series & forecasting (Prophet, N-BEATS, TFT)**

Track R — Research (2 weeks):
- Weeks α+β · Paper-reading protocol, reproduce 3 seminal papers (Transformer, **mini-Chinchilla — 3 model sizes on TinyStories fitting exponents**, DPO), reading lists per company

**Design decision:** all six track-week MDX articles and reference projects ship. Learners pick two of the three tracks; every choice is available. The historical fallback for an incomplete research track was not used.

### 9.1 W24 Capstone
Spec §8 lists 3 capstone tracks (Research Engineering, Applied ML, LLM Product / Agentic). Each has multiple options. Ship: rubric templates in `capstone/track-{research,applied-ml,llm-product}/RUBRIC.md`, no full projects (the learner builds these).

### 9.2 W25 Interview prep
Spec §6.4 has a 7-day protocol (D1 coding refresh → D7 outreach). Ship this as `week-25-interview-prep.mdx` with day-by-day breakdown + linked resources.

**Estimated size:** ~8-10 MDX (6 track weeks + capstone rubrics + W25) + 6 track projects. Tag `v1.0.0` — first full-book release.

---

## 10. Established patterns (follow these)

### 10.1 Per-week MDX shape

The exact template of Weeks 1-4. Use these as reference; do not invent new sections.

```mdx
---
week: <N>
part: <1..6>
slug: "week-XX-<kebab-slug>"
title: "<Title>"
hook: "<one-sentence pitch>"
hours: 20
computeTier: "green" | "yellow" | "red"
difficulty: <1..5>
prereqSlugs: ["week-XX-...", ...]              # at least prior week
referenceProject: "projects/week-XX-<name>"
accentVar: "--accent-p<N>"                     # p1=Month1, p2=Month2, ... (see spec §2.3)
---

import Hook from '../../components/content/Hook.astro'
import Intuition from '../../components/content/Intuition.astro'
import MathBlock from '../../components/content/MathBlock.astro'
import CompanyLens from '../../components/content/CompanyLens.astro'
import ReferenceProject from '../../components/content/ReferenceProject.astro'
import Assignments from '../../components/content/Assignments.astro'
import InterviewDrill from '../../components/content/InterviewDrill.astro'
import FurtherReading from '../../components/content/FurtherReading.astro'
import KeyTakeaways from '../../components/content/KeyTakeaways.astro'
import IntuitionCallout from '../../components/callouts/Intuition.astro'
import Gotcha from '../../components/callouts/Gotcha.astro'
import { MicroRecall } from '../../components/interactive/MicroRecall'

<Hook>...</Hook>
<Intuition>
  ... intuition prose ...
  <IntuitionCallout>...</IntuitionCallout>
  ... four moves prose ...
  <MicroRecall questions={[{id, prompt, choices, correct, explain}]} client:visible />
</Intuition>

## The math, derived
### 1. ...
<MathBlock latex="..." />
### 2. ...

<Gotcha>...</Gotcha>

## The code, from scratch
```language
...code...
```
Full implementation: `projects/week-XX-<name>/`.

<CompanyLens
  topic="<topic>"
  tldr={{ openai: "...", anthropic: "...", deepmind: "...", meta: "...", xai: "...", deepseek: "...", qwen: "..." }}
  whyDiffer="..."
  sources={{ openai: { paper, blog }, ... 7 entries ... }}
  caseStudy="..."
  interviewAngle={{ openai: "...", ... 7 entries ... }}
/>

<ReferenceProject path="projects/week-XX-<name>" name="<name>" hours="6-8 hrs">
  Description.
</ReferenceProject>

<Assignments>
- **Warmup (30 min):** ...
- **Build (2-3 hrs):** ...
- **Challenge (3+ hrs):** ...
</Assignments>

<InterviewDrill role="research-eng|applied-ml|llm-product" time="20 min">
**"..."**
Rubric:
- baseline signal
- senior signal
- staff signal
Worked solution: `projects/week-XX-<name>/SOLUTION_NOTES.md`.
</InterviewDrill>

<FurtherReading>
- **...** — one line description.
</FurtherReading>

<KeyTakeaways>
- 5 bullet takeaways.
</KeyTakeaways>
```

### 10.2 CompanyLens all-7-slugs rule

The `<CompanyLens>` component asserts at build time that `tldr`, `sources`, and `interviewAngle` each contain all 7 slugs. Missing one fails the build with `CompanyLens tldr: missing entries for [qwen]` or similar. See `apps/book/src/lib/companies.ts:33`.

The 7 slugs in FIXED order: `openai, anthropic, deepmind, meta, xai, deepseek, qwen`.

### 10.3 Uniform Python project shape

```
projects/week-XX-<name>/
├── README.md                (what it is, how to run, public API)
├── SOLUTION_NOTES.md        (gotchas, pitfalls, "what surprised me")
├── COMPUTE.md               (tier, time, budget, cloud-runbook if 🔴)
├── pyproject.toml           (see template below)
├── src/<pkg_snake_case>/
│   ├── __init__.py          (re-export public API)
│   └── ...modules...
├── tests/                   (pytest files, name pattern `test_*.py`)
├── notebooks/               (percent-format .py files, convert to .ipynb with jupytext)
└── assignments/
    ├── warmup.md            (30 min)
    ├── build.md             (2-3 hrs)
    └── challenge.md         (3+ hrs)
```

**pyproject.toml template:**
```toml
[project]
name = "<pkg-name>"                # kebab-case, matches src dir hyphenated
version = "0.1.0"
description = "Week X — <topic>"
requires-python = ">=3.13"
dependencies = [<pinned>]

[project.optional-dependencies]
dev = ["pytest>=8.3", "hypothesis>=6.112", "ruff>=0.7", "pyright>=1.1"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
addopts = "-v --strict-markers"
testpaths = ["tests"]
```

### 10.4 TDD policy

- **Correctness code (autograd, numerical, ML impl):** RED → GREEN. Write the failing test first.
- **Content (MDX prose, README):** no test required — the build + link-check catches these.
- **Framework wiring (Astro config, Vite):** verified by `pnpm --filter book run build` returning exit 0.
- **Every task ends with a commit.** Never batch multiple tasks in one commit.

### 10.5 Attribution snapshot test

`apps/book/tests/company-attribution.spec.ts` greps every `apps/book/src/content/companies/*.mdx` for forbidden strings:

```js
const FORBIDDEN_STRINGS = [
  ['DPO must be attributed to Stanford, not Meta',            /\bDPO\b(?:(?!\bnot\s+Meta\b).){0,60}\bMeta\b/i],
  ['A2A/ADK must be attributed to Google Cloud, not DeepMind',/\b(?:A2A|ADK)\b(?:(?!\bnot\s+DeepMind\b).){0,80}\bDeepMind\b/i],
  ['No Colossus peer-reviewed paper exists',                  /Colossus\s+paper|Colossus\s+arXiv/i],
  ['Chinchilla is DeepMind (not OpenAI/Anthropic)',           /Chinchilla(?:(?!\bnot\s+(?:OpenAI|Anthropic)\b).){0,60}\b(?:OpenAI|Anthropic)\b/i],
  ['Age of AI has 3 authors (Kissinger, Schmidt, Huttenlocher)', /Age of AI[^.]{0,80}Kissinger[^.]{0,80}Schmidt(?!.{0,80}Huttenlocher)/i],
]
```

Any new MDX in `apps/book/src/content/companies/` that trips a pattern will fail this test. The negation regexes allow disclaimers ("A2A + ADK are Google Cloud, NOT DeepMind") — the pattern matches WRONG attribution ONLY.

### 10.6 Test-running commands

```bash
# Ensure Node 24 is on PATH (session default is often Node 20):
export PATH="$HOME/.nvm/versions/node/v24.21.0/bin:$PATH"

# Python — per project (uv workspace doesn't hoist dev deps):
for d in projects/week-*/; do
  cd "$d" && uv run --extra dev pytest 2>&1 | tail -3 && cd -
done

# JS unit (viz):
pnpm --filter @core-ai/viz test

# JS unit (book — vitest under jsdom):
pnpm --filter book run test:unit

# JS integration (Playwright — uses astro dev + workers:1):
pnpm --filter book test

# Build:
pnpm --filter book run build     # includes pagefind postbuild → dist/client/pagefind/

# Link check (weekly cron):
pnpm run check-links
```

### 10.7 Git conventions

- Branch: `plan-N-<slug>` (e.g. `plan-4-weeks-5-8`).
- Merge back with `git merge --ff-only` (no merge commits).
- Commit prefix: `feat(wXX):`, `feat(book):`, `test(...)`, `fix(...)`, `chore:`, `docs:`, `ci:`.
- Tag milestones: `v0.<plan>.0-<milestone>` (e.g. `v0.4.0-week08`).
- Delete feature branches after merge: `git branch -d <name>`.

### 10.8 Node version dance

The repo pins Node 24, but many sessions default to Node 20. Every Node command in this repo needs:

```bash
export PATH="$HOME/.nvm/versions/node/v24.21.0/bin:$PATH"
```

or:
```bash
export NVM_DIR="$HOME/.nvm"; [ -s "$NVM_DIR/nvm.sh" ] && source "$NVM_DIR/nvm.sh"; nvm use 24
```

Astro will refuse to run under Node 20 with the message: `Node.js v20.19.5 is not supported by Astro! Please upgrade Node.js to a supported version: ">=22.12.0"`.

---

## 11. Attribution rules (verbatim, spec §5.3)

**Every one of these is checked by `company-attribution.spec.ts`. Violating one is a Critical bug.**

- **DPO** — Rafailov, Sharma, Mitchell, Ermon, Manning, Finn 2023. **Stanford.** Never "DPO — Meta." (Meta hired the co-authors; that's the connection, not authorship.)
- **A2A + ADK** — Google **Cloud** products. Never "A2A — DeepMind" or "ADK — DeepMind."
- **Attention Is All You Need** — Vaswani et al. 2017, Google Brain pre-merger. Defensible under today's Google DeepMind org.
- **Chinchilla** — Hoffmann et al. 2022, DeepMind. Note the Epoch AI replication which caught systematic errors.
- **Colossus** (xAI cluster) — documented via NVIDIA/Spectrum-X blog + HPCwire / DataCenterDynamics reporting. **No peer-reviewed publication exists.** Never cite a "Colossus paper" or "Colossus arXiv."
- **DDIA (Designing Data-Intensive Applications)** — 2E, March 2026, Kleppmann + **Riccomini**. The 2E co-author matters.
- **Age of AI** — Kissinger, Schmidt, **Huttenlocher** (3 authors). Never omit the third.
- **7 first-class labs, exact display names:** OpenAI · Anthropic · Google DeepMind · Meta AI (FAIR) · xAI · DeepSeek · Alibaba Qwen. Always in this order in CompanyLens tables.

---

## 12. Known issues + deferred items

### 12.1 Reduced motion — fixed
`ScrollReveal` now progressively enhances visible SSR content with a small IntersectionObserver/CSS transition. Reduced-motion changes disconnect the observer and remove transforms; no-JavaScript content stays readable. Three real browser tests replace the deferred fixmes.

### 12.2 WeeklyQuiz persistence — fixed
Actual globally unique question IDs key persisted `{reps,ease,interval,dueDate}` alongside answer statistics. Legacy valid records remain readable; malformed rows are discarded. Final-question feedback is visible before the summary. LocalStorage remains browser-local and may be unavailable in private/quota-limited contexts.

### 12.3 /companies/compare outside Pagefind
SSR routes intentionally remain outside the static Pagefind index. Every company profile now links to `/companies/compare`, as do the matrix/sidebar. Search discovers the static profiles rather than indexing arbitrary query-dependent comparisons.

### 12.4 Development search — fixed
`apps/book/pagefind-dev.mjs` serves only the existing built index under `/pagefind`, with path traversal protection. Build before running dev search. Browser tests cover actual index results and unavailable-index recovery. The native dialog handles focus/Escape, and stale asynchronous results are rejected.

### 12.5 15 Minor findings from Plan 2 review
Listed exhaustively in the final message of the Plan 2 execution session. Highlights:
- Anthropic and Qwen share 🟠 emoji (M1)
- Meta emoji 🟢 doesn't match its blue tint (M2)
- Anthropic's "Machines of Loving Grace" listed under `books:` — it's an essay (M3)
- `assertAll7` now rejects null/undefined and inherited entries (M4 fixed).
- Some Playwright tests do only filesystem I/O — should move to vitest (M5, M15)

The remaining emoji/reading-category cosmetics and test-runner organization are nonblocking maintenance, not missing curriculum.

### 12.6 Strict Python typing — still open
The full strict Pyright audit reported thousands of diagnostics; it is not passing. Strict mode remains configured and runs as a visibly advisory CI step. Runtime pytest and Ruff are required gates. Annotate/stub packages incrementally and restore a blocking strict baseline after proving it clean. See `docs/VALIDATION.md`; do not report all static typing as green.

---

## 13. Rulings log (decisions made and why)

Ledgered rulings from Plans 2 + 3 that future operators should understand rather than re-litigate:

| Ruling | Reason | Cost if wrong |
|---|---|---|
| Added `packages/viz/vitest.config.ts` (not in Plan 2) | TSX tests needed Preact JSX runtime | Minimal config drift |
| Preact `compat: true` + Vite alias react→preact/compat + `ssr.noExternal: ['motion','framer-motion']` | Motion v13 pulls framer-motion which imports React | +5-6 KB gzipped bundle |
| CompanyMatrix iterates 1..25 explicitly, not the weeks collection | 25 columns must render before Plans 3-8 fill in weeks | Cell mismatch when weeks come in |
| Reordered Task 14 (Week 1 CompanyLens rewrite) before Tasks 9-13 in Plan 2 | Parity assertion tripped the build | Same delivery, different order |
| Pagefind output at `dist/client/` (not `dist/`) | Vercel adapter output location | 404 on `/pagefind/pagefind.js` |
| SearchDialog uses `new URL(...).href` at runtime | Rollup can't resolve `/pagefind/pagefind.js` at build | TypeScript can't statically check module shape |
| Playwright uses `astro dev` (not `astro preview`) | Vercel adapter blocks `astro preview` | None — dev SSR is same code |
| Playwright `workers: 1` | HMR races between workers on shared dev server | ~30% slower suite runtime |
| Historical motion fixmes replaced in Plan 8 | Visible-by-default SSR plus an observer avoids hidden reduced-motion content | Three browser regressions now cover reduction, preference changes and no-JS |
| Refined attribution regexes to exclude negated disclaimers | Original patterns flagged correct disclaiming text | A "not-Y" style wrong attribution slips through |
| Node 24 via nvm PATH-prepend when needed | Session default is Node 20; astro needs ≥22 | Dev-loop friction; CI already runs Node 24 |
| Duplicate DeepSeek `week: 15` seededAngles merged into one entry | Object-key collision in `CompanyMatrix.astro` dropped one silently | Users see fewer angles in matrix |
| Company MDX bodies made sparse (frontmatter is authoritative) | `CompanyLayout` already renders every section from frontmatter; body sections were dupes | None visible; content mildly less discoverable |
| Skipped `vercel --prod` and remote deployment | User instruction: they'll set up new Vercel account | None — user runs when ready |
| W5–W8 CompanyLens copy uses public-evidence caveats and practice prompts | The handoff outline included undocumented internal implementation claims; the spec forbids inventing facts | Slightly less assertive copy, but attribution-safe and teachable |
| W7 acceptance uses iris petal length/width for ARI ≥ 0.7 | The all-feature k-means claim is not a reliable acceptance fixture; the separable view is documented in tests and lesson text | Does not imply unsupervised clustering recovers all iris species from every feature view |
| W5 closed form uses augmented `lstsq` and scales the L2 penalty by `n` | Mean-loss gradient descent and an unscaled normal-equation penalty optimize different objectives; `lstsq` also handles rank deficiency | A few extra lines, with safer educational numerics |
| W8 permutation importance uses a validation holdout and mixed-type leakage can be `unavailable` | Measuring importance on fit rows is optimistic, and correlation cannot inspect categorical/object data | More honest report shape; domain audits remain necessary |
| Added `http://localhost:4321/$s` to link-check ignores | Historical plan contains a shell-loop URL artifact, not a clickable link | One explicit ignore entry in the checker |
| Week 9 uses a deterministic seven-segment MNIST-shaped fixture instead of downloading MNIST | CI and offline notebooks must not depend on a network or a large dataset while still exercising 28×28 tensor shapes | Learners must replace the fixture with real MNIST in the build assignment |
| Week 10 keeps CIFAR training optional and tests use fake data | M-series local accuracy depends on hardware and runtime; correctness tests should be fast and offline | Accuracy claims stay honest; the notebook is a starting point, not a benchmark guarantee |
| Week 11 uses a tiny in-memory corpus and exposes LSTM/GRU attention | A reproducible attention bridge is more useful than a network-dependent Shakespeare download | Less literary data, but deterministic shape/gradient/attention tests |
| Week 12 includes Q-learning, REINFORCE, and a tested PPO-Clip objective without generative models | The spec places the RL primer in W12 and moves VAE/GAN/diffusion to W17 | PPO is explicitly conceptual here; a complete PPO trainer remains future scope |
| Plan 6 uses offline toy fixtures and optional integrations for LLM/post-training/multimodal work | CI must be deterministic and cloud credentials/model downloads are unavailable; toy tests still expose tensor, objective, retrieval, and checkpoint contracts | Toy metrics cannot support production quality, scaling, or company-internal architecture claims |
| W13/W14/W17 difficulty values are capped at schema maximum 5 | The content scope is advanced, but `content.config.ts` intentionally restricts the display scale to 1–5 | The lessons communicate advanced scope through topic/compute labels without breaking the schema |
| W15a/W15b and L/P/R routes have explicit navigation labels | Numeric week IDs are not unique lesson IDs; last-write-wins maps lose alternatives | 25 roadmap columns retain distinct 15a/15b and all track links |
| W14 checkpoint stores model configuration and tokenizer metadata; W15a DPO and W15b GRPO-like updates remain toy objectives | Tokenizer/model compatibility and attribution boundaries are correctness contracts, not optional prose | More metadata and caveats, but safer continuation by future operators |
| W15b received the standard `ReferenceProject` component after the first build | The initial MDX had the project prose but omitted the rendered project card, caught by the new route smoke test | One small content fix; full lesson anatomy is now present |
| Correctness review expanded W13/W15a/W15b tests and fixed objective/state/router bugs | Initial worker tests proved the happy path but missed causal target shifts, reference-policy mutation, grouped precision, and top-1 router gradients | More local tests and clearer toy boundaries; no production-scale claim |

---

## 14. How to execute a plan (the recipe that works)

This is what shipped Plans 2 + 3. Follow it verbatim.

1. **Read** the spec (`docs/superpowers/specs/2026-09-22-core-ai-book-design.md`) sections relevant to your plan's weeks. Do NOT skim.
2. **Branch:** `git checkout -b plan-N-<slug>` off `plan-1-foundation`.
3. **Ledger:** create `.superpowers/sdd/<plan-basename>/progress.md` with `# SDD ledger — plan: docs/superpowers/plans/YYYY-MM-DD-<name>.md` as the first line. Log any rulings there.
4. **Extend company `seededAngles` first** — the matrix depends on this. One commit.
5. **Per week (repeat 4 times for Plan 4, 5 times for Plan 6, etc):**
   1. Scaffold Python project per §10.3.
   2. Write pyproject + one source module + one failing test.
   3. `uv run --extra dev pytest` — see it fail.
   4. Implement.
   5. Test passes.
   6. Add remaining modules + tests. Full green.
   7. README + SOLUTION_NOTES + COMPUTE + 3 assignment files + 1 notebook.
   8. Commit: `feat(w0X): <project> — <one-line summary>`.
   9. Write MDX per §10.1. Include ALL 7 CompanyLens slugs.
   10. `pnpm --filter book run build` — see the new week's route prerender.
   11. Commit: `feat(w0X): <topic> MDX with 8-part anatomy + 7-company Lens`.
6. **Cross-plan tests:**
   1. Extend `apps/book/tests/company-lens-parity.spec.ts` to include the new weeks in `WEEKS` array.
   2. Add `apps/book/tests/weeks-N-M-render.spec.ts` — smoke test each new route.
   3. Add `apps/book/tests/katex-weeks-N-M.spec.ts` — assert `<span class="katex` present in built HTML.
   4. Commit: `test(weeks N-M): parity + smoke + KaTeX SSR on new week routes`.
7. **Full green suite** — run all 4 test commands from §10.6. Any failure blocks tag + merge.
8. **Milestone commit:** `git commit --allow-empty -m "chore: plan N complete — <headline>"`.
9. **Tag:** `git tag -a vX.Y.0-<milestone> -m "..."`.
10. **Merge:** `git checkout plan-1-foundation && git merge --ff-only plan-N-<slug> && git branch -d plan-N-<slug>`.
11. **Push:** `git push origin plan-1-foundation && git push origin --tags`.
12. **Delete plan workspace:** `rm -rf .superpowers/sdd/<plan-basename>/` — git history is the record.
13. **Update this HANDOFF.md** — bump the "current state" section, add ledgered rulings to §13, note any new deferred items in §12.

If a build or test fails: **STOP.** Investigate. Prefer root-cause fixes over patching over. If you edit a plan step, ledger the ruling in the workspace `progress.md`.

---

## 15. Escalation & stop conditions

Four things stop you. Do NOT proceed past these without explicit user consent:

1. **Irreversible / destructive git operations** — `git reset --hard`, `git push --force`, `git branch -D`. Ask first, always.
2. **Security-sensitive changes** — auth, secrets, env vars, dependency downgrades.
3. **External side effects** — `vercel --prod`, publishing packages, creating public GitHub content, sending Slack/email.
4. **Plan so broken every path forward is a guess** — spec contradicts itself, dependencies unavailable, etc. Surface the ambiguity and ask.

Anything else — including "the plan says X but Y is clearly better" — is a ruling. Make the call, ledger it in `.superpowers/sdd/<plan>/progress.md`, and continue.

---

## 16. Contact + follow-ups

- **User:** Tharun (`sritharun242004` on GitHub).
- **Prior operator:** Claude Opus 4.7 (Anthropic), 2026-09-27 → 2026-09-28.
- **Preferred cadence:** the user asks for autonomy ("go", "continue", "in sequence"). Take it, ship, report. Ask only for the 4 stop conditions in §15.
- **Deferred deployment:** user is setting up a new Vercel account. Do not deploy under existing accounts.
- **Remaining maintenance:** strict Python typing audit, optional external/GPU integration validation, routine source freshness and cosmetic profile cleanup. Motion, quiz persistence and dev-search regressions are fixed; compare remains SSR with discoverable profile links. See §12 and `docs/VALIDATION.md`.

---

## 17. If you're stuck

- Read the spec. Every question this repo answers is answered there first.
- Read this HANDOFF twice.
- Read `docs/superpowers/plans/2026-09-27-core-ai-platform-features.md` and `docs/superpowers/plans/2026-09-28-core-ai-weeks-2-4.md` — they demonstrate the full pattern for both a platform plan and a content plan.
- Grep the codebase for a similar existing case (`grep -r "CompanyLens" apps/book/src`). Almost every pattern already has 4 examples.
- If truly stuck, write to the user and describe what you're stuck on. Don't guess.

---

Good luck. Ship.

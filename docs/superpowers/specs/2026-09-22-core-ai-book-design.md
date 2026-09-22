# Core AI — From Foundation to Frontier
## Design Spec (v2 — verified)

| | |
|---|---|
| **Status** | Draft v2 — verified against 4 parallel research agents; pending final user review |
| **Author** | Tharun (learner) + Claude (design) |
| **Date** | 2026-09-22 |
| **Path** | `Core-AI/docs/superpowers/specs/2026-09-22-core-ai-book-design.md` |
| **Next step** | User review → `writing-plans` skill → phased implementation plan |
| **Changes from v1** | 24 → 25 weeks (W15 split honestly); tech-stack version bumps; Google/Google-Cloud attributions fixed; supporting cast expanded; per-week compute-tier callouts added; expanded interview-prep coverage |

---

## 1. Purpose & Intent

Build the best possible self-study curriculum for a full-stack engineer with zero prior AI/ML background to reach competitive interview level for AI-engineer roles at frontier labs and big tech in ~6 months of intensive study.

The deliverable is a beautiful, responsive, HTML-based learning platform — an editorial textbook plus 25 real Python reference projects, threaded with company-comparison sidebars, interview drills, and spaced-recall quizzes — that the learner uses daily as their primary study material.

## 2. Learner Profile
- **Role:** Full-stack engineer, strong at code.
- **AI baseline:** True beginner. Math and ML concepts are new territory.
- **Style:** Heavy visual explanation — diagrams, math derivations, flowcharts, graphs — over prose.
- **Ambition:** Interview at all three role types (research eng, applied ML, LLM product) at major labs.

## 3. Constraints (updated)
- **Time budget:** 20–22 hrs/week × 25 weeks ≈ **500–550 hours** (v1 said 480; verification exposed real over-scoping in W15, so the honest total is one week larger).
- **Solo learner:** No cohort, no live instructor. Material must be self-driving.
- **Language split:** JS/TS for the site (Astro), Python for all ML content.
- **Hardware:** Apple Silicon MacBook (M-series) as primary; **cloud GPUs required for W16, W17, W18, W20, W23** — budget $150–300 total ($20–50 for RunPod/Modal per session).
- **Faithfully & responsibly:** No hand-waving. Every concept covered with primary sources, working code, and honest gap-acknowledgment.

## 4. Non-Goals
- Not a webapp with backend, auth, or per-user server state.
- Not a video course.
- Not a live cohort platform.
- Not a marketplace or paid product.

## 5. Success Criteria
1. Implement transformers, RLHF/DPO, and a small RAG-agent system **from scratch in PyTorch**.
2. Pass an ML-fundamentals oral exam on any of the 25 weekly topics without notes.
3. Articulate how OpenAI / Anthropic / Google DeepMind / Meta / xAI / Mistral / DeepSeek-Qwen differ across ≥ 6 specific technical axes.
4. Ship 3 portfolio-quality artifacts (capstone + 2 top-tier assignments) to GitHub.
5. Complete ≥ 50 interview-style drills across the 4 surfaces (coding, system design, fundamentals, behavioral).

---

## Section 1 — Curriculum Roadmap (25 weeks · 6.25 months)

**Format:** 6 Parts × 4–5 weeks. Each week is ~20 hrs. Compute tier flagged per week: 🟢 local M-series · 🟡 M-series stretch or optional cloud · 🔴 cloud GPU required.

### Month 1 — Mathematical & Programming Foundations
- **W1** 🟢 Linear algebra — vectors, matrices, eigenvalues, SVD
- **W2** 🟢 Calculus for ML — derivatives, gradients, chain rule, Jacobians, Hessians
- **W3** 🟢 Probability & statistics — distributions, Bayes, MLE/MAP, entropy, KL, cross-entropy
- **W4** 🟢 Python for ML + information theory — NumPy, pandas, matplotlib, PyTorch basics

### Month 2 — Classical Machine Learning
- **W5** 🟢 Linear/logistic regression, gradient descent from scratch
- **W6** 🟢 Trees, ensembles, XGBoost, SVMs
- **W7** 🟢 Unsupervised — k-means, PCA, GMM, t-SNE, UMAP
- **W8** 🟢 Model evaluation, bias/variance, CV, feature engineering, leakage

### Month 3 — Deep Learning Foundations
- **W9** 🟢 Neural nets from scratch + backprop derivation
- **W10** 🟡 CNNs, ResNets, augmentation, transfer learning *(90% on CIFAR-10 in ~1 hr on MPS is realistic; 93%+ wants a cloud GPU)*
- **W11** 🟢 RNNs, LSTMs, seq2seq, the attention mechanism
- **W12** 🟢 Optimizers (SGD/Adam/AdamW), regularization, **RL primer (MDPs, Q-learning, policy gradient, PPO)** — VAE/GAN/diffusion moved to W17

### Month 4 — Transformers, LLMs, Generative AI *(now 5 weeks — W15 split for honest scope)*
- **W13** 🟡 Attention Is All You Need — build transformer from scratch in PyTorch; **plus post-transformer landscape (SSMs / Mamba / Jamba / Griffin) — compare, contrast, know when each wins**
- **W14** 🟡 Tokenization, embeddings, **positional encoding depth (RoPE / ALiBi / YaRN, long-context tricks — needle-in-haystack, ring-attention preview)**, pre-training vs post-training paradigms, weights & checkpoints
- **W15a** 🔴 **Post-training foundations** — SFT, RLHF, **DPO + its family (KTO, IPO, ORPO, SimPO)**, RLAIF, Constitutional AI, **LoRA / QLoRA / DoRA / PEFT**, synthetic-data pipelines (self-instruct, distillation) *(model size: Llama-3.2-1B or Qwen-0.5B on M-series; ≥ 7B → cloud)*
- **W15b** 🔴 **Advanced post-training** — LLM families (GPT / Claude / Llama / Mistral / Qwen / DeepSeek / Gemma / Kimi / GLM), scaling laws, **MoE deep dive (routing, load balancing, DeepSeek-V3 mechanics)**, **reasoning models & test-time compute (o-series / R1)**, **GRPO / RLVR / verifiers / DualPipe**
- **W17** 🟡 Multimodal — CLIP, ViT, **VAE/GAN/diffusion (moved from W12) + world/video models (Sora 2, Veo 3.1, Genie, Cosmos, JEPA)**, audio + **voice-native pipelines (Whisper + streaming ASR + TTS: ElevenLabs, OpenAI/Google real-time)**, **RAG deep dive + named variants (GraphRAG, HyDE, ColBERT, late-interaction, hybrid retrieval)**, semantic search, re-ranking, retrieval metrics (Recall@k, MRR, NDCG, MAP), knowledge-graphs intro

### Month 5 — AI Systems & Production
- **W18** 🔴 Distributed training (data / tensor / pipeline parallelism, FSDP, DeepSpeed, Megatron, **ring / context-parallel attention**), **GPU/TPU internals + CUDA / Triton kernel intro**
- **W19** 🔴 Inference engines (vLLM, TGI, TensorRT-LLM, Triton, llama.cpp, **plus Groq / Cerebras / SambaNova as alternative silicon**), quantization (INT8/INT4, GPTQ, AWQ, KV-cache quant), **KV cache paging**, speculative decoding, latency vs throughput (TTFT, TPOT, batching), **prompt & context caching (Anthropic/OpenAI)**
- **W20** 🟡 **MLOps + Evaluation Infrastructure** — data pipelines, feature stores, monitoring, CI/CD for ML, synthetic-data pipelines; **LLM-as-judge, MT-Bench, HELM, MMLU-Pro, SWE-Bench, Inspect AI, Harbor, trajectory & agent evals, contamination detection**
- **W21** 🟡 AI safety & alignment — RLHF deep, Constitutional AI, red-teaming batteries, **interpretability (logit lens, probing, sparse autoencoders / dictionary learning — Anthropic Circuits methods)**, safe-agent foundations

### Month 6 — Specialization + Capstone + Interview Prep

**W22–W23 = 4 weeks total.** Learner picks **2 of 3 tracks**:

**Track L — LLM Product / Agentic (2 weeks)**
- Week α · Agents & agentic AI — architectures (ReAct/Reflexion/Plan-and-Execute), tool use, planning, memory, long-horizon tasks, multi-agent, **function calling / structured output / tool schemas / parallel tools**, harmless + headless agent patterns
- Week β · Protocols & frameworks — **MCP, A2A, ADK (Google Cloud products, not DeepMind)**, LangGraph, CrewAI, AutoGen, OpenAI Agents SDK, Knowledge-Graph RAG, prompt & context engineering, **agent evals (SWE-Bench Verified, Inspect AI, trajectory metrics)**, **cost engineering / LLM economics** (batch pricing, model routing, distillation-for-cost, cache economics)

**Track P — Applied ML (2 weeks)**
- Week α · Recsys, ranking, ads ML, search ranking, retrieval at scale, GNNs (recsys, fraud, molecules, social)
- Week β · Learning-to-rank + neural rerankers + A/B testing + causal inference basics, **time series & forecasting (Prophet, N-BEATS, TFT)**

**Track R — Research (2 weeks)**
- Weeks α+β · Paper-reading protocol, reproduce 3 seminal papers (Transformer, **mini-Chinchilla — 3 model sizes on TinyStories fitting exponents**, DPO), reading lists per company

- **W24** Capstone — one large end-to-end project (Section 8)
- **W25** Interview prep — coding + ML system design + ML fundamentals oral + behavioral + resume/portfolio + outreach

### Companion track (threaded, not weekly)
- `/extras/paper-reading-protocol` — how to read 3 papers/week
- `/extras/tech-writing` — postmortems, design docs, blog posts
- `/extras/glossary` — every term, cross-linked
- `/extras/math-primer` — bootstrap refresher

**Total effective coverage: 500+ concepts, 7 labs compared 25 ways, 25 real reference projects, ~50 interview drills, 1 capstone.**

---

## Section 2 — Platform Architecture & Design System

### 2.1 Mode
**Read** with a splash of **Experience** so it's worth returning to daily for 6+ months. References: Bartosz Ciechanowski, *The Illustrated Transformer*, distill.pub (archive — on hiatus since 2021), Anthropic Docs, Stripe Docs.

### 2.2 Tech stack *(verified & version-corrected)*

| Layer | Choice | Notes |
|---|---|---|
| Framework | **Astro 6** + MDX | v6 (Mar 2026) ships Live Content Collections + first-party Fonts API |
| Styling | Tailwind CSS v4 (Lightning CSS) | v4.3.x stable |
| Fonts | **Astro 6 Fonts API** managing Fraunces / Inter / JetBrains Mono / Newsreader | Replaces manual `public/fonts` self-hosting |
| Math | KaTeX (SSR) | Zero client cost |
| Diagrams | Mermaid + hand-authored SVG + D3 + Observable Plot + **D2 (for W18/W19 arch diagrams)** | D2 gives better output for complex architecture; keep Mermaid for GitHub-renderable |
| Code blocks | Shiki + Expressive Code | VS Code engine, zero-JS |
| Interactive islands | Preact + Signals | 3-4 KB baseline; TC39 Stage 1 |
| Motion | **Motion (motion/react v12)** + GSAP ScrollTrigger | Framer Motion renamed to Motion in 2025 |
| A11y primitives | **Base UI** (shadcn/ui default since July 2026) | Radix development slowed; Base UI is the successor by the same authors |
| Color scales | Radix Colors | Unaffected, still best-in-class |
| Prose | @tailwindcss/typography | Long-form substrate |
| Components (cherry-picked) | shadcn/ui (re-themed, Base-UI-backed) | Copy-paste, we own the code |
| Icons | Lucide + Vercel Geist Icons | Editorial, clean |
| Search | Pagefind | Client-side, zero backend |
| Python islands | Pyodide inline | Small snippets in-browser |
| Deployment | Vercel + Astro adapter | Static + edge, free previews |
| Optional lint pre-pass | **Oxlint** | 2× Biome for large repos |

### 2.3 Design tokens
- **Type scale:** 1.250 (Major Third) — 12/14/16/20/25/31/39/49/61 px
- **Body:** 18px / 1.7 line-height / 68ch max — reading first
- **Spacing:** 4pt grid (4/8/12/16/24/32/48/64/96)
- **Radius:** 2/4/8/12 — quiet, never bubbly
- **Color:** neutral-first + one accent per Part (indigo → teal → violet → amber → emerald → rust)
- **Dark mode:** first-class, `data-theme` attribute swap
- **Motion:** scroll-linked reveals only, `prefers-reduced-motion` respected
- **A11y:** WCAG AA minimum; math + diagrams have text alternatives

### 2.4 Visual world — "Editorial Textbook"
Bartosz Ciechanowski meets *Kinfolk*. Wide margins, serif long-form body, oversized display headings, hand-authored SVG diagrams that scroll-animate, generous whitespace. Warm, quiet, deeply readable. One accent color per Part.

---

## Section 3 — Weekly Lesson Anatomy

Every one of the 25 weeks uses the identical 8-part page structure. Consistency > novelty for daily study.

1. **The Hook** (5 min) — one striking question. No jargon.
2. **Intuition First** (30 min) — visual explanation with hand-authored SVGs, real-world analogy, before any math. → 💡 Micro-recall (2 MCQ)
3. **The Math, Derived** (2–3 hrs) — full derivation, every symbol defined, colored terms consistent across the book (query = indigo, key = teal, value = rust), KaTeX-rendered, worked numeric example. → 💡 Micro-recall (2 fill-in)
4. **The Code, From Scratch** (3–4 hrs) — pure NumPy / raw PyTorch, for-loop first then vectorized, line-by-line commentary, Jupyter/Colab link.
5. **Company Lens 🔍** (30 min) — the 5-block template (TL;DR table, philosophy, primary sources, case study, interview angle).
6. **Reference Project 🧪** (6–8 hrs) — a real cloneable repo in `projects/week-XX/`.
7. **Assignments** (4–6 hrs) — Warm-up (30 min) → Build (2–3 hrs) → Challenge (3+ hrs), each with hidden reference solutions.
8. **Interview Drill 🎯** (1 hr) — one problem tagged by role and surface, with rubric and reveal-on-click worked solution. → 🧠 Weekly synthesis quiz (10 questions)

### Recurring visual devices
- Colored math terms (consistent palette book-wide)
- Margin annotations (right on desktop, inline on mobile)
- Scroll-linked diagrams
- 5 semantic callouts: 💡 Intuition · ⚠️ Gotcha · 🔬 Deep dive (collapsible) · 🎯 Interview · 📎 Company lens
- Prereq breadcrumbs — every week links back to specific prior weeks
- **Compute-tier badge** at top of each week (🟢 / 🟡 / 🔴)

---

## Section 4 — Company Lens System *(now 7 companies)*

### 4.1 The 7 fixed companies
Every week compares the same 7 labs so comparison compounds:

- 🟣 **OpenAI** — GPT / o-series (successor system cards through Sep 2026), Sora 2, Codex; RLHF pioneers, product-first
- 🟠 **Anthropic** — Claude, Constitutional AI, MCP, Circuits interpretability, safety-first
- 🔵 **Google DeepMind** — Gemini 2.5 / 3, AlphaFold/Zero/Go, TPU stack; **note: A2A + ADK are Google Cloud products, not DeepMind**
- 🟢 **Meta AI (FAIR)** — Llama 3.x, PyTorch, SAM-2, ImageBind, open weights, **DPO co-authors (with Stanford — Rafailov et al.)**
- ⚫ **xAI** — Grok, Colossus 200k-GPU cluster (documented via NVIDIA/Spectrum-X blog + HPCwire reporting; no peer-reviewed papers)
- 🟡 **DeepSeek** — V3 / R1 / V3.5, GRPO, DualPipe, MoE reasoning frontier
- 🟠 **Alibaba Qwen** — Qwen 3, open-weight frontier; often groups with Kimi K2, GLM-4.5, MiniMax-M1 in the Chinese open-frontier bucket

**Supporting cast** (mentioned when relevant, no full profile):
- **Silicon & inference:** NVIDIA, Groq, Cerebras, SambaNova
- **Training infra:** Databricks/Mosaic (MPT, DBRX), Together AI, Fireworks AI, Modal, Replicate
- **Enterprise LLM:** Cohere (Command-R, Embed v3, Rerank 3), Mistral (post-2025 enterprise pivot: Le Chat, Codestral), Microsoft (Phi, Copilot)
- **Media:** Runway Gen-4, Pika, Luma, HeyGen (video) · ElevenLabs, Suno, Udio (audio) · Midjourney, Black Forest Labs (Flux), Stability (image)
- **Ecosystem:** Hugging Face, Perplexity, Cursor, ByteDance (Doubao/Seedream/Seed-VL)

### 4.2 The 5-block Company Lens template (per week)
1. **TL;DR table** — 1 row per company × N-column comparison on that week's axis
2. **Why they differ** — 2 short paragraphs on the philosophical split
3. **Primary sources** — 1 landmark paper + 1 blog post per company
4. **Case study** — one real story (e.g., "The Sydney incident" for W21)
5. **Interview angle** — what THIS company actually asks on this topic

### 4.3 `/companies` synthesis route
- `/companies` — 7 × 25 grid matrix landing
- `/companies/<slug>` — full profile per company (founding, leadership, model timeline, papers, philosophy, interview loop, books, notable engineers, incidents)
- `/companies/compare?companies=…&axis=…` — pick any subset + any axis

**By Week 25: 7 × 25 = 175 comparison cells internalized.**

---

## Section 5 — Reference Projects + Assignments + Reading Lists

### 5.1 The 25 reference projects *(compute tier flagged; renumbered)*

| Week | Project | Focus | Tier |
|---|---|---|:-:|
| W1 | `linalg-lab/` | NumPy vector-space + SVD image compression | 🟢 |
| W2 | `micrograd/` | Karpathy-style autodiff engine (~200 LOC) | 🟢 |
| W3 | `prob-lab/` | Monte Carlo, Bayes, entropy visualizations | 🟢 |
| W4 | `numpy-vs-pytorch/` | Same 5 problems, three ways | 🟢 |
| W5 | `linreg-from-scratch/` | GD + logistic, no sklearn | 🟢 |
| W6 | `xgboost-kaggle/` | Full Kaggle submission pipeline | 🟢 |
| W7 | `unsupervised-viz/` | PCA + k-means + GMM + t-SNE | 🟢 |
| W8 | `ml-eval-suite/` | CV, learning curves, calibration, importance | 🟢 |
| W9 | `mini-torch/` | Extend micrograd → MLP on MNIST | 🟢 |
| W10 | `cifar-resnet/` | ResNet-18 from scratch → **~90% on MPS in ~1 hr; 93%+ needs cloud/MLX overnight** | 🟡 |
| W11 | `char-rnn-attention/` | Char-level LSTM + attention on Shakespeare | 🟢 |
| W12 | `rl-gridworld/` | Q-learning + policy gradient on gridworld | 🟢 |
| W13 | `nano-gpt-ssm/` | **~1-2M-param GPT on Shakespeare on any M-series (30 min); 10M variant on M3+ Max w/ MLX**; side-by-side with a tiny Mamba/SSM baseline | 🟡 |
| W14 | `mini-bpe + pretrain/` | BPE from scratch + TinyStories pretraining (3–5 hrs single GPU) | 🟡 |
| W15a | `sft-lora-dpo-lab/` | SFT + LoRA + DPO on **Llama-3.2-1B (local) or Llama-3-8B (cloud)** with DPO variants comparison | 🔴 |
| W15b | `moe-and-reasoning/` | Tiny MoE from scratch + GRPO training loop on GSM8K subset | 🔴 |
| W17 | `mini-rag-multimodal/` | End-to-end RAG (GraphRAG + HyDE + ColBERT + hybrid) with evals + tiny VAE/diffusion demo + voice pipeline | 🟡 |
| W18 | `fsdp-ring-lab/` | Multi-GPU FSDP + ZeRO + ring-attention long-context | 🔴 |
| W19 | `vllm-benchmark/` | vLLM vs HF, KV tricks, spec decoding, Groq/Cerebras notes | 🔴 |
| W20 | `evals-mlops-pipeline/` | MLflow + DVC + W&B + drift + **eval harness: MT-Bench + Inspect AI + LLM-judge** | 🟡 |
| W21 | `alignment-lab/` | **DPO-based** tiny RLHF loop + 20-prompt red-team + logit-lens/tiny-SAE interp probe | 🟡 |
| W22–23 · L·α | `agents-lab/` | Track L: ReAct + Reflexion + Plan-and-Execute + long-horizon | 🟡 |
| W22–23 · L·β | `mcp-a2a-adk-lab/` | Track L: MCP server/client + A2A demo + Google Cloud ADK sample + SWE-Bench harness | 🟡 |
| W22–23 · P·α | `two-tower-recsys/` | Track P: MovieLens + GNN recsys variant | 🟡 |
| W22–23 · P·β | `learning-to-rank-timeseries/` | Track P: LambdaMART + neural rerankers + N-BEATS/TFT forecasting | 🟡 |
| W22–23 · R | `paper-repro-lab/` | Track R: Reproduce Transformer + mini-Chinchilla + DPO | 🔴 |

*(Learner picks 2 of {L, P, R} — building 3–4 projects across W22–W23.)*

### 5.2 Uniform per-project shape
```
projects/week-XX-<name>/
├── README.md
├── SOLUTION_NOTES.md
├── COMPUTE.md           ← required cloud steps + budget estimate
├── src/
├── tests/
├── notebooks/
├── data/
├── configs/
├── assignments/{warmup,build,challenge}.md
└── solutions/  (hidden, spoiler-blocked in book)
```

### 5.3 Books & readings *(attributions corrected & sources expanded)*

**Universal foundations:**
- *Deep Learning* — Goodfellow, Bengio, Courville
- *Pattern Recognition and Machine Learning* — Bishop
- *The Elements of Statistical Learning* — Hastie, Tibshirani, Friedman
- *Reinforcement Learning: An Introduction (2E)* — Sutton & Barto
- *Dive Into Deep Learning* (d2l.ai) — Zhang et al.
- *Speech and Language Processing (3rd ed. draft, Jan 2025)* — Jurafsky & Martin *(free at web.stanford.edu/~jurafsky/slp3)*
- *Hands-On ML w/ Scikit-Learn, Keras & TensorFlow (3E, 2022)* — Géron
- *Build a Large Language Model from Scratch* (Manning, 2024) — Raschka
- *Designing Machine Learning Systems* (O'Reilly, 2022) — Huyen
- *AI Engineering* (O'Reilly, 2024/25) — Huyen
- *Deep Learning for Coders with fastai & PyTorch* — Howard & Gugger

**Interview-focused:**
- *Machine Learning System Design Interview* (ByteByteGo, 2023) — Aminian & Xu
- *Deep Learning Interviews* (arXiv:2201.00650) — Kashani & Ivry
- *Ace the Data Science Interview* — Singh & Huo
- ***Designing Data-Intensive Applications 2E*** (March 2026) — Kleppmann + **Riccomini**

**Cross-lab research references (not owned by any single company):**
- **Attention Is All You Need** (Vaswani et al. 2017) — Google Brain (pre 2023 Google Brain + DeepMind merger; defensible under today's Google DeepMind org)
- **DPO** (Rafailov, Sharma, Mitchell, Ermon, Manning, Finn 2023) — **Stanford** (not Meta)
- **Chinchilla** (Hoffmann et al. 2022) — DeepMind (see Epoch AI's replication which caught systematic errors in the original)

**Blogs & courses (evergreen):**
- Karpathy — YouTube "Zero to Hero" + **nanoGPT + llm.c repos**
- Jay Alammar — Illustrated Transformer / Illustrated GPT-2 / Illustrated Diffusion
- distill.pub archive (hiatus since July 2021)
- Chip Huyen — huyenchip.com
- Sebastian Raschka — Substack + *LLMs in 2026* PyCon keynote
- Lilian Weng — lilianweng.github.io
- **Nathan Lambert — Interconnects** (best contemporary post-training / RL Substack)
- Full-Stack Deep Learning course
- **Stanford CS336 Spring 2026** (Language Modeling from Scratch)

**🟣 OpenAI**
- Papers: GPT-1/2/3/4 tech reports · InstructGPT · Whisper · CLIP · DALL·E · Codex · GPT-4 System Card · Sora 2 tech doc · o1/o3 system cards + **current OpenAI flagship system card (Sep 2026)**
- Blog: openai.com/research
- Books: *The Age of AI* — Kissinger, Schmidt, **Huttenlocher** (3rd author, missed in v1); *Genius Makers* — Metz

**🟠 Anthropic**
- Papers: Constitutional AI (arXiv:2212.08073) · Claude model cards · Sleeper Agents (arXiv:2401.05566) · Toy Models of Superposition · Scaling Monosemanticity · MCP spec · **Anthropic RSP v3.0 (Feb 2026)**
- Blogs: anthropic.com/research · transformer-circuits.pub (Chris Olah — 20+ posts on interpretability)
- Books/essays: Amodei "Machines of Loving Grace"

**🔵 Google DeepMind**
- Papers: Attention Is All You Need (see cross-lab note) · AlphaGo/Zero/Fold · Chinchilla · Gopher · **Gemini 2.5 / 3 tech reports** · Sparrow · Gemma technical reports · PaLM · Flamingo · Griffin · Mixture-of-Depths
- Docs: JAX + Flax · TPU whitepapers · **A2A + ADK docs (Google Cloud, not DeepMind)**
- Videos: Hassabis 2024 Nobel Prize (Chemistry) lecture (nobelprize.org)

**🟢 Meta AI (FAIR)**
- Papers: Llama 1/2/3/3.1/3.2/3.3 · SAM/SAM-2 · ImageBind · DINOv2 · Chameleon · CodeLlama · MegaBlocks
- Docs: PyTorch documentation (canonical study text)
- Position: LeCun — *A Path Towards Autonomous Machine Intelligence* (2022)

**⚫ xAI**
- Reports: Grok 1/1.5/2/3 technical documents · Colossus cluster (**NVIDIA/Spectrum-X blog + HPCwire/DataCenterDynamics reporting — no peer-reviewed papers**)
- Blog: x.ai/blog
- Reading: Isaacson *Elon Musk* (2023, chapters on xAI); Sutton *The Bitter Lesson* (2019)

**🟡 DeepSeek + Chinese open frontier**
- Papers: DeepSeek-V2/V3/V3.5 · DeepSeek-Coder · DeepSeek-R1 (GRPO invention, arXiv:2501.12948) · Qwen 2.5/3 · Kimi K2 · GLM-4.5 · MiniMax-M1
- Blogs: deepseek.com/research · Qwen blog

---

## Section 6 — Interview Prep Integration *(expanded per real 2026 loop data)*

### 6.1 Four interview surfaces
1. **ML Coding** — includes: pure DSA (Meta 2 rounds), PyTorch impl, **AI-assisted / LLM-collaborator coding (Meta/xAI/Anthropic)**, **CodeSignal 90-min take-homes (Anthropic)**, **xAI 4-hr full-stack take-home**, **ML debugging round (OpenAI)**
2. **ML System Design** — "Design YouTube recs", "serve Claude at 100k QPS"
3. **ML Fundamentals (oral)** — derivations, "why does BatchNorm break at inference?"
4. **Behavioral + project deep-dive + values screens** — Anthropic RSP alignment, DeepMind ethics

### 6.2 Threaded practice (expanded)
- **Weeks 9 → 23:** one tagged interview drill per week (**15 seeded drills**)
- **Every Company Lens block:** 2–4 tagged interview angles (**~100 angle-only drills** across 25 weeks × 7 companies)
- **Coding companion track:** **20–30 additional Leetcode-ML problems** in `/interview/coding-set/` (was under-weighted in v1)
- **AI-assisted coding practice** (2–3 tasks): extend a real codebase using Claude/Cursor while an interviewer watches
- **Take-home rehearsals** (2 timed): one 90-min (Anthropic-style) + one 4-hr (xAI-style)
- **ML debugging drills** (3): reproduce a real training failure and diagnose it

### 6.3 Role distribution
- `[research-eng]` ~6 seeded drills (math-heavy, from-scratch impls)
- `[applied-ml]` ~5 seeded drills (recsys/ranking/MLOps)
- `[llm-product]` ~5 seeded drills (RAG/agents/evals)

### 6.4 Week 25 protocol *(expanded to 7 days, more surfaces)*
- **D1** Coding refresh (top 50 ML-flavored Leetcode + PyTorch impl drills)
- **D2** ML fundamentals oral (timed 20-min drills, out loud, recorded)
- **D3** System design (2 full mock designs: recsys + RAG-at-scale)
- **D4** Company deep dives (top 3 targets: loop structure, signature questions, recent papers)
- **D5** Behavioral + STAR (8 stories × ~40 competencies) + **values screens (Anthropic RSP, DeepMind ethics)** + **recruiter-call rehearsal**
- **D6** Full mock loop (4 rounds, recorded, reviewed) + **1 AI-assisted coding round**
- **D7** **Resume + LinkedIn optimization (half day) + cold-email/hiring-manager outreach playbook + comp research (levels.fyi, PPU, RSU) + portfolio-site polish + EU timezone/scheduling logistics for Mistral / DeepMind London**

### 6.5 Company-specific loop guides
Each `/companies/<name>` page includes a **loop guide**: public interview structure, known bar-raiser patterns from Blind/Glassdoor/exponent.com/igotanoffer/interviewing.io, top ~10 signature questions.

### 6.6 `/interview` dashboard route
Preact island with `localStorage`-backed state:
- Progress by surface, role, company (7 rows now)
- **Weak-spot heatmap** — surfaces failed spaced-recall concepts and serves targeted drills
- Streak, problems solved, mock loops
- **Caveat:** no cross-device sync (localStorage is single-device)

---

## Section 7 — Monorepo Structure *(toolchain versions corrected)*

### 7.1 Toolchain (locked)
- **JS/TS pkg mgr:** pnpm (workspaces)
- **JS/TS lint+format:** Biome (+ optional **Oxlint** pre-pass for speed on large repos)
- **Task runner:** Turborepo (Rust rewrite; still the default JS/TS monorepo runner)
- **Python pkg mgr:** uv (Astral)
- **Python lint+format:** ruff
- **Python typecheck:** pyright (or **ty**/Astral once GA)
- **Python test:** pytest + hypothesis
- **Node:** **24 LTS** (Node 22 exits Active LTS Sep 2026)
- **Python:** **3.13** (3.12 in security-only mode by late 2026; 3.14 optional for template strings)
- **Deploy:** Vercel (Astro adapter)
- **CI:** GitHub Actions
- **Git hooks:** lefthook

### 7.2 Top-level layout
```
core-ai-book/
├── README.md
├── package.json           (pnpm workspace root)
├── pnpm-workspace.yaml
├── pyproject.toml         (uv workspace root)
├── uv.lock
├── turbo.json
├── biome.json
├── oxlint.json            (optional)
├── ruff.toml
├── .github/workflows/{book,projects,ci,link-check}.yml
├── apps/
│   └── book/              (Astro 6 site)
├── packages/
│   ├── ui/
│   ├── content-schema/
│   ├── theme/
│   └── viz/
├── projects/              (25 Python reference repos)
├── solutions/
├── shared/
│   ├── data/
│   ├── viz_py/
│   └── evals/            (LLM-judge, MT-Bench harness, Inspect AI integration)
├── docs/superpowers/specs/
├── scripts/
│   ├── new-week.mjs           (scaffold book+project+solution+COMPUTE.md)
│   ├── check-links.mjs        (verify arXiv/blog links resolve — CI)
│   └── generate-companies.mjs (build /companies matrix from frontmatter)
└── capstone/
    ├── track-research/
    ├── track-applied-ml/
    └── track-llm-product/
```

### 7.3 `apps/book/` internals
```
apps/book/
├── astro.config.mjs             (Astro 6 + integrations)
├── tailwind.config.ts
├── src/
│   ├── content/                 (Astro Content Collections + Live Collections for /companies)
│   │   ├── config.ts            (Zod schemas)
│   │   ├── weeks/*.mdx          (25 weeks)
│   │   ├── companies/*.mdx      (7 companies)
│   │   └── extras/*.mdx         (glossary, math-primer, paper-reading, tech-writing, how-to-study)
│   ├── components/              (layout, content, callouts, viz, quiz, dashboard)
│   ├── layouts/                 (WeekLayout, CompanyLayout, PageLayout)
│   ├── pages/                   (index, weeks/[slug], companies/[slug], companies/compare, interview/index, interview/coding-set, glossary, how-to-study, 404)
│   ├── lib/                     (content, progress, quiz)
│   └── styles/                  (global, prose, math, tokens)
└── public/                      (og, favicon — fonts now handled by Astro 6 Fonts API)
```

---

## Section 8 — Capstone Specification

Week **24** is one big project (~40 hrs). Three tracks — pick one aligned with your W22-23 mini-books.

### 8.1 Track A — Research Engineering
> Reproduce or extend a frontier result and write the paper.

Options: **mini-Chinchilla** (3 model sizes on TinyStories, fit scaling exponents, honest error bars) · **mini-GRPO** (from-scratch training on GSM8K) · **mini-MoE** (dense vs sparse at matched active params) · **mini test-time-compute** (tiny reasoning model).

**Rubric (5 × 20 pts):** correctness · rigor (ablations, error bars, seeds) · code quality · writeup (4-page arXiv format) · novelty (one extension).

### 8.2 Track B — Applied ML
> Ship a production-grade ranking or recsys system.

Options: YouTube-scale 4-stage recommender on MovieLens 25M · IEEE-CIS fraud with GNN+GBM hybrid · ad ranking + A/B simulator · M5 hierarchical forecasting with N-BEATS/TFT.

**Rubric:** production shape · metrics rigor · systems thinking (latency, cost) · experiments (offline + A/B sim + counterfactual) · design doc.

### 8.3 Track C — LLM Product / Agentic
> Ship an agentic system a user could actually rely on.

Options: MCP-native research agent with KG-backed citations · multi-agent codebase reviewer w/ SWE-Bench eval · fine-tuned + RAG hybrid for a specialized domain · voice-native agent (Whisper → LLM → TTS, sub-second TTFT).

**Rubric:** product quality (real UI, error paths, guardrails) · eval rigor (task evals + LLM-judge + red-team + regression) · cost & latency (measured & optimized) · agent design · writeup + demo video.

### 8.4 Uniform deliverable shape
```
capstone/track-<X>/<name>/
├── README.md
├── WRITEUP.md
├── DEMO.md
├── COMPUTE.md
├── src/
├── tests/
├── evals/
├── configs/
├── notebooks/
└── SELF_RUBRIC.md
```

### 8.5 Week 24 day-by-day
- D1 Scope + kickoff writeup
- D2–3 Architecture + skeleton (repo up, tests failing)
- D4–5 Core implementation
- D6 Evals + polish
- D7 Writeup + demo + README

---

## Open Questions / Risks / Assumptions

- **Assumption:** M-series MacBook + a small cloud budget ($150–300 across W15a, W15b, W17, W18, W19, W22-R, W24) covers the whole curriculum.
- **Assumption:** Learner has 20-22 hrs/week for 25 weeks, not 20 flat.
- **Risk:** Still tight for **originating** novel research at frontier-lab bar; delivers **interview-competent** across all three tracks.
- **Risk:** Fast-moving field — every named model / paper / product must be reviewed quarterly. `scripts/check-links.mjs` runs weekly in CI.
- **Open question (for user):** Publish Week 1 *before* the whole book is complete so the learner can start day one? (Recommended: yes.)

---

## Verification Log

Four independent research agents cross-checked v1 of this spec against public 2025-2026 primary sources (papers, blog posts, interview loops from Exponent / igotanoffer / interviewing.io / Blind / Glassdoor, tool release notes). Summary of applied fixes:

- **Curriculum:** split W15 → W15a + W15b; folded evals into W20 explicitly; moved VAE/GAN/diffusion from W12 → W17; added DPO variants (KTO/IPO/ORPO/SimPO), SSMs/Mamba, named RAG variants, voice/world-model families, ring/context-parallel attention, sparse-autoencoder interp depth.
- **Tech stack:** Astro 5 → 6; Framer Motion → Motion v12; Radix Primitives → Base UI (Radix Colors stays); Node 22 → 24 LTS; Python 3.12 → 3.13; added Astro Fonts API + optional Oxlint + D2 for complex diagrams.
- **Companies & sources:** DPO attributed to Stanford (not Meta); A2A + ADK attributed to Google Cloud (not DeepMind); Attention Is All You Need flagged as Google Brain pre-merger; DDIA cited as 2E (Kleppmann + Riccomini); *Age of AI* now correctly credits Huttenlocher; Colossus cited via reporting (no peer-reviewed papers exist); Alibaba Qwen promoted to 7th first-class lab; supporting cast expanded with Cohere, Groq/Cerebras/SambaNova, Databricks/Mosaic, Together/Fireworks/Modal, Runway/Pika/Luma/ElevenLabs/Flux/Suno, ByteDance; added Karpathy nanoGPT/llm.c repos and Nathan Lambert's Interconnects.
- **Feasibility:** W13 nanoGPT claim qualified (M3+ Max w/ MLX or 1-2M param variant on any M-series); W10 CIFAR accuracy honest (90% local, 93% cloud/overnight); W15 model size stated (1B local, 7B cloud); W21 alignment-lab uses DPO not PPO + logit-lens/tiny-SAE instead of full interp; mini-Chinchilla reframed with honest scope; per-week compute-tier badges added.
- **Interview:** added coding companion track (20–30 more Leetcode-ML), AI-assisted coding drills, take-home rehearsals (Anthropic 90-min + xAI 4-hr), ML debugging drills, resume/LinkedIn/cold-email/comp negotiation on D7 of prep week.

---

## Definition of Done

- [x] All 8 sections drafted
- [x] Self-review pass
- [x] External verification (4 parallel agents)
- [x] Fixes applied inline
- [ ] **User approves this v2 document**
- [ ] `writing-plans` skill invoked to produce phased implementation plan

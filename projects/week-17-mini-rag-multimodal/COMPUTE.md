# Compute card — yellow tier

## Local path: **$0 required**

This reference project is designed to run offline on a laptop CPU. The corpus is eight
short documents held in memory; retrieval is NumPy vector arithmetic; the VAE and one
 denoising step use tiny arrays. No checkpoint, dataset, audio service, network access,
GPU, account, or API key is needed.

| Activity | Local expectation |
| --- | --- |
| 37+ unit tests | well under a minute on a current laptop |
| notebook smoke run | seconds, CPU, deterministic |
| memory | a few tens of MB plus Python |
| required cost | **$0** |
| artifact | terminal output and optional local JSON/plots |

The yellow label means that a learner may swap in a real encoder, reranker, or
multimodal model for an extension. The baseline itself is green in practice. Do not
add a hosted service to the default path.

## Optional cloud guidance (not required and not run by tests)

If you replace the fixtures with a named open checkpoint, use an ephemeral CPU or small
GPU notebook, set a hard wall-clock and budget cap, cache the checkpoint, and delete the
runtime when finished. A small GPU is useful for comparing a real image/text encoder or
batching a larger local corpus; it is not needed for the contracts in this repository.
Check current provider pricing before starting—prices and free quotas change. Never put
provider keys in this repository. Record provider, region, instance type, checkpoint
identifier, dataset license, seed, package lock, wall-clock, and total spend in an
experiment note. Cloud results are not comparable to the toy fixture unless the query
set, labels, k, candidate budget, and evaluation protocol are held constant.

A safe extension plan is:

1. keep the deterministic tests and fixture as a regression suite;
2. run a local/offline dry run before enabling a provider SDK;
3. cap spend and time outside the code's default path;
4. compare Recall@k, MRR, NDCG, MAP, latency, and failure cases—not just one answer;
5. remove credentials and shut down the resource immediately after the run.

The voice protocol in this project remains an interface seam. A hosted ASR/TTS demo is
optional research infrastructure, not evidence that the offline adapter recognizes
speech or that a company's production voice architecture works this way.

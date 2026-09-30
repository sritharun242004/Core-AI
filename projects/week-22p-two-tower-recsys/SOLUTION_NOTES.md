# Solution notes — spoilers

## Warmup

For ranking `[9, 2]` and positive set `{2, 3, 4}`, Recall@2 is **1/3**, not 1/2 or 1. Binary DCG is `1/log2(3) = .63093`; ideal DCG is `1 + 1/log2(3) = 1.63093`, so NDCG@2 is approximately **.38685**. Items absent from candidates are still relevant. For graded labels use `2**grade - 1`; ideal ordering uses the full truth mapping. Empty relevance contributes zero (state this convention; some benchmarks skip it).

## Build

Split unique user/item events before constructing graph edges or popularity counts. Every user's final timestamp must be strictly later than all that user's fit times. Reject repeated edges rather than letting the same item leak across train and holdout. Real repeated-consumption tasks need a different stated contract, and a global serving-time cutoff must be designed separately.

BPR is `softplus(-(u·i_pos - u·i_neg))`; its margin derivative is `-sigmoid(-margin)`, so gradient descent increases positive relative to negative. Both embeddings must change. Do not form negatives from all-data labels: excluding heldout positives from negative sampling would reveal future information. Training-only unobserved negatives may include future likes, which is a genuine implicit-feedback limitation.

For a bipartite graph, `A_norm = D^-1/2 A D^-1/2`. The graph representation is `(E + A_norm E)/2`. No feature transform or nonlinearity is required for this particular LightGCN-style variant. Zero-degree nodes remain finite because inverse degrees are clamped, but their random initial vectors are not a cold-start solution. Tests inspect missing holdout edges, exact normalized neighbor values, and nonzero gradients.

## Challenge / interview worked solution

Start a search/ads recommendation design with the candidate universe, user intent, exposure logs and latency budget. Retrieve a broad set using a dot-product index, rank a smaller set with richer interaction features, then apply policy/diversity constraints. ID towers precompute catalog embeddings but cannot model arbitrary user-item cross features independently. A cross-encoder can do so later at higher cost. Version towers and indexes together; monitor stale items, index recall, end-to-end relevance and serving latency separately.

Popularity is essential: it can exploit shared taste while a sparse ID model overfits. A graph may improve sparse neighborhoods but can amplify popularity and homophily, transfer sensitive correlations, and leak future edges. Graph temporal audits apply equally to fraud networks, molecular scaffold splits and social links; a random edge split is not universally meaningful.

Do not claim that offline NDCG gain proves causal lift. Logging exposure and position create selection bias; ads additionally need delayed conversion, calibration, bid/budget constraints and user guardrails. Retrieval cannot rank an item it never retrieved. Report conditional reranker NDCG separately from end-to-end quality, and carry the full relevant set into Recall denominators.

## Deliberate boundaries

No heldout hyperparameter selection, real MovieLens performance claim, scalable ANN implementation, content cold-start encoder, full LightGCN reproduction or uncertainty claim from a single seed. The build assignment creates validation/test splits before a parameter sweep; the challenge requests multi-seed user-bootstrap uncertainty. Those extensions are learner work, not secretly completed features.

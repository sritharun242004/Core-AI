# Solution notes — spoilers

## Pairwise ranking and LambdaRank

For one high/low pair with margin zero, RankNet loss is log 2 and score derivatives are [-.5,+.5]. Relevance ties produce no preferred pair. Queries must never be compared against one another: score offsets across queries are not identified by a within-query objective.

For two grades [1,0], current tied scores resolve by original row order. IDCG@2=1, and the absolute change from swapping is `1 - 1/log2(3) = .369070`. Multiplying the logistic derivatives yields lambdas about `[-.184535,+.184535]`. The implementation divides by the number of preferred pairs within each query and then by the total query count. Equal labels produce zero; all-zero relevance produces zero ideal gain and zero metric.

Swap weights are recomputed at current ranks but detached during backprop. Finite differences agree away from rank boundaries. Do not claim a smooth derivative of NDCG: the ranking itself is discrete. Candidate eligibility cannot shrink the truth set; masked relevant rows still hurt Recall and ideal DCG. Tied model scores are broken by stable input order, not labels.

LambdaMART combines ranking lambdas with boosted trees; this reference trains a neural MLP instead. A genuine optional extension could use LightGBM `LGBMRanker(objective='lambdarank')` with query-contiguous rows and `group` lengths, validation query groups, pinned versions and an explicit dependency. That adapter is **not implemented**, so no LambdaMART result is reported. A neural reranker may also be a cross-encoder; our numeric MLP is not one.

## Experiments and causal interpretation

Control [1,3] has mean 2 and sample variance 2; treatment [4,6] has mean 5 and variance 2. ATE=3, SE=sqrt(2/2+2/2)=sqrt(2), nominal normal 95% CI approximately [.2282,5.7718]. With only two units per arm, this illustrates the formula, not well-calibrated small-sample inference. The production choice might be a design-based/randomization interval, a t approximation or a bootstrap justified for the design.

Randomize the causal unit, not every page view. One user receiving both policies creates contamination. Cluster assignment requires cluster-aware analysis; social-network spillovers violate no-interference assumptions. Never condition analysis on clicking after treatment or exclude dissatisfied treated users. Predeclare the primary metric, guardrails, horizon and minimum detectable effect; peeking requires a valid sequential procedure, not repeated fixed-horizon p-values.

SRM compares observed arm counts to planned allocation, not necessarily 50/50. Counts [900,100] under a 50/50 plan are a strong red flag. Investigate assignment, logging and eligibility before interpreting a treatment delta. A nonsignificant SRM result does not prove the experiment is unbiased.

Observational comparisons need assumptions beyond this API: conditional exchangeability/no unmeasured confounding, overlap/positivity and consistency. Inverse propensity weighting and doubly robust estimators do not manufacture overlap or remove unmeasured confounding. No such estimator is implemented. This lab estimates unadjusted randomized-unit differences only.

## Forecasting

At origin t, fit preprocessing and all training windows on indices strictly below t. Inputs end at t-1 and scored targets are t through t+h-1. Historical training windows may overlap; each target must still lie inside the prefix. Do not create windows over the entire series and randomly split them, fit a global scaler, interpolate through future values, or select epochs with final-test errors.

Seasonal naive with last season [4,5,6] and horizon five gives [4,5,6,4,5]. It is not a constant-last-value baseline. On seasonal signals it can be difficult to beat; failure to beat it is a meaningful result.

Each neural block computes `hidden = MLP(residual)`, `backcast = B(hidden)`, `forecast = F(hidden)`. Subtract backcast before the next block and sum forecasts. The first backcast receives gradients through downstream residual blocks. The final backcast has no downstream use and hence no forecast-loss gradient; tests assert this instead of inventing a reconstruction loss from the paper. Dense learned heads are generic bases, not interpretable seasonal/trend components.

Rolling reports contain point MAE and raw arrays. They do not estimate forecast uncertainty. Use a past-only calibration period, interval coverage/width diagnostics and dependence-aware validation for any quantile/conformal extension. Overlapping horizons are not independent replicates; neither are repeated origins necessarily independent with nonoverlapping targets.

## Interview worked response

A higher offline NDCG, an imprecise A/B delta and a better forecast answer three different questions. Freeze query groups and eligibility first; inspect candidate recall, exposure/position bias and metric denominators. Audit experiment assignment, SRM, independent units, missingness and CI against a business-relevant threshold. For demand planning, roll forecasting origins, compare seasonal naive and prevent future covariate leakage. Known future calendar features differ from future observed sales/weather. Do not launch merely because one of three statistics looks favorable.

Prophet is an additive trend/seasonality/holiday model with configurable changepoints; TFT combines variable selection, gated recurrent/local processing and temporal attention for multi-horizon forecasting with static/known-future/observed inputs. Neither is implemented here. Their names are comparison targets, not brands to put on an unrelated toy network.

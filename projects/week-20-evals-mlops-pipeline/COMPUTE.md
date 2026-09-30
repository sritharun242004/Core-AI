# Compute — yellow

The reference suite and notebook run on CPU in seconds, with no credentials or downloads and $0 provider spend. A production evaluation may be expensive even without training: two swapped judge calls per pair doubles judge requests; repeated seeds and long trajectories multiply it further.

Before optional external evaluation, pin provider/model revision, dataset/license, maximum cases, input/output token caps and a dollar stop condition. Example arithmetic only: 100 cases × 2 calls × 2,000 input tokens = 400,000 input tokens; add output tokens and retries using the provider's current price, not an assumed free tier. Check current prices manually; this repository does not call providers.

Use local MLflow file tracking and W&B offline mode first. If renting a GPU (for example RunPod L4/A10 class), check a current quote, cap the run at one hour, budget a provisional $1–5 plus storage and egress, and stop on budget exhaustion. This is an estimate, not a measured requirement or guaranteed quote. Export only nonsensitive metrics; stop the instance, terminate the pod, delete unwanted volumes, revoke temporary credentials, and verify the billing dashboard. Never let a nightly evaluation silently provision compute.

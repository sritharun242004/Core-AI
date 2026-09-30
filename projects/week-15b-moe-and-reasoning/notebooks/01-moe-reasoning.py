# %% [markdown]
# Week 15b — tiny MoE and verifiable reasoning
#
# Offline percent-format notebook. It uses in-memory tensors only: no corpus,
# checkpoint, tokenizer, network request, or cloud service.

# %%
import torch
from moe_reasoning_lab import (
    ArithmeticPolicy,
    TinyMoE,
    TinyMoEClassifier,
    arithmetic_reward,
    group_relative_advantages,
    grpo_step,
    make_toy_supervised_data,
    test_time_majority_vote,
    train_supervised_moe,
)

# %% [markdown]
# ## 1. Route tokens to experts
#
# The router produces all-expert probabilities, then chooses top-k experts and
# renormalizes selected weights for k > 1. Top-1 retains its selected
# all-expert probability so the task loss can train the router. Capacity marks
# assignments that cannot fit; inspect the mask instead of silently pretending
# every assignment ran.

# %%
torch.manual_seed(7)
moe = TinyMoE(d_model=8, hidden_dim=16, num_experts=4, top_k=2, capacity_factor=1.0)
hidden = torch.randn(2, 5, 8)
output, aux_loss = moe(hidden, return_aux=True)
routing = moe.last_routing
print("output shape:", tuple(output.shape))
print("top-k index shape:", tuple(routing.topk_indices.shape))
print("selected weight sums:", routing.topk_weights.sum(-1)[:5])
print("capacity:", routing.capacity)
print("expert loads:", routing.expert_loads.tolist())
print("dropped assignments:", int((~routing.dispatch_mask).sum()))
print("finite auxiliary loss:", bool(torch.isfinite(aux_loss)))

# %% [markdown]
# ## 2. Toy supervised MoE optimization
#
# This is a fixed, linearly separable synthetic batch. A decreasing loss says
# the model, gradient, and auxiliary term are connected; it is not a language
# benchmark or a scaling result.

# %%
features, labels = make_toy_supervised_data(n_samples=32, seed=7)
classifier = TinyMoEClassifier(input_dim=2, d_model=16, hidden_dim=24, num_experts=3)
training = train_supervised_moe(classifier, features, labels, steps=45, learning_rate=0.03)
print("initial/final supervised loss:", training.losses[0], training.losses[-1])
print("final expert loads:", classifier.moe.last_routing.expert_loads.tolist())

# %% [markdown]
# ## 3. Arithmetic verifier and grouped relative rewards
#
# A verifier is useful when correctness is mechanically checkable. The local
# parser accepts a deliberately narrow binary arithmetic format and compares
# the final number; it is not a general evaluator.

# %%
questions = ["What is 17 + 25?", "What is 9 * 6?"]
responses = [["42", "41", "The answer is 42", "0"], ["54", "53", "54", "54"]]
rewards = torch.tensor(
    [
        [arithmetic_reward(question, answer) for answer in group]
        for question, group in zip(questions, responses, strict=True)
    ]
)
print("rewards:", rewards)
print("relative advantages:", group_relative_advantages(rewards))

# %% [markdown]
# ## 4. Minimal GRPO-like grouped update
#
# The implementation centers/scales rewards within each prompt group and uses
# a clipped policy-ratio objective. This is a local teaching implementation,
# not a claim to reproduce undocumented production training.

# %%
policy = ArithmeticPolicy(answer_vocab_size=64)
optimizer = torch.optim.AdamW(policy.parameters(), lr=0.02)
prompts = torch.tensor([[17, 25, 0], [9, 6, 2]])
actions = torch.tensor([[42, 41, 42, 0], [54, 53, 54, 54]])
old_log_probs = torch.log_softmax(policy(prompts), dim=-1).gather(1, actions)
answer_rewards = torch.tensor(
    [[arithmetic_reward("What is 17 + 25?", int(action)) for action in group] for group in actions]
)
loss = grpo_step(policy, optimizer, prompts, actions, answer_rewards, old_log_probs)
print("sampled answer ids:", actions.tolist())
print("rewards:", answer_rewards.tolist())
print("one grouped update loss:", loss)

# %% [markdown]
# ## 5. Test-time compute: majority vote
#
# Majority vote spends more verifier/model calls and only helps when correct
# candidates are frequent. The fixed seed makes this demonstration repeatable.


# %%
def noisy_arithmetic_sampler(question, rng):
    answer = {"What is 17 + 25?": 42, "What is 9 * 6?": 54}[question]
    return answer if rng.random() < 0.65 else answer + rng.choice([-1, 1])


winner, samples = test_time_majority_vote(
    "What is 17 + 25?", noisy_arithmetic_sampler, num_samples=9, seed=7
)
print("samples:", samples)
print("majority vote:", winner)

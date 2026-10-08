# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownLambdaType=false, reportMissingParameterType=false, reportUnknownParameterType=false, reportCallIssue=false, reportArgumentType=false, reportOptionalMemberAccess=false, reportAttributeAccessIssue=false
# %% [markdown]
# Mechanism reproduction, not WMT BLEU reproduction. No external data.
# %%
import torch
from transformer_repro import attention, train_fixture

torch.set_num_threads(1)
query = torch.zeros(1, 3, 4)
values = torch.tensor([[[1.0], [3.0], [8.0]]])
outputs, weights = attention(query, query, values)
print("Causal prefix averages:", outputs.squeeze(-1).tolist())
print("Attention:", weights.tolist())
for seed in (1, 2, 3):
    history = train_fixture(seed=seed)
    print({"seed": seed, "initial_fixture_loss": history[0], "final_fixture_loss": history[-1]})
# %% [markdown]
# Report pre-norm, GELU, learned positions and decoder-only scope as deviations
# from Vaswani et al. 2017. An overfit periodic corpus measures no translation skill.

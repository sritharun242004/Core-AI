# %% [markdown]
# # Track L alpha: decisions, observations, feedback, and a shared budget
# Execute from the repository root with project-local PYTHONPATH (see README).
# No API keys, model downloads, arbitrary code execution, or network tools.

# %%
from agents_lab import (
    Action,
    Budget,
    Call,
    PlanAndExecute,
    ReAct,
    Reflexion,
    ScriptedModel,
    fixture_registry,
)

registry = fixture_registry()
model = ScriptedModel(
    [
        Action(
            calls=(Call("lookup", {"key": "guide"}), Call("lookup", {"key": "hours"})),
            parallel=True,
        ),
        Action(calls=(Call("add", {"a": 8, "b": 12}),)),
        Action(final="The public guide recommends 20 study hours."),
    ]
)
report = ReAct(registry, model, Budget(4, 3)).run("Read the guide and add the study hours")
assert report.status == "completed"
assert (report.steps, report.calls) == (3, 3)
for event in report.trajectory:
    print(event.kind, event.data)

# %% [markdown]
# Reflexion-style retry: a trusted verifier provides feedback, not hidden reasoning.
# Both attempts draw from the same budget. A failed answer is never a verified success.

# %%
reflexion = Reflexion(
    registry, ScriptedModel([Action(final="19"), Action(final="20")]), Budget(2, 0)
)
verified = reflexion.run(
    "8+12",
    verifier=lambda answer: answer == "20",
    feedback="Recheck the arithmetic; the target is 20.",
)
assert verified.status == "completed"
assert verified.steps == 2
print("reflexion:", verified.status, verified.answer)

# %%
planner = ScriptedModel(
    [
        Action(calls=(Call("lookup", {"key": "guide"}),)),
        Action(final="Public records only"),
        Action(calls=(Call("add", {"a": 8, "b": 12}),)),
        Action(final="20"),
    ],
    plan=("Read policy", "Compute total"),
)
planned = PlanAndExecute(registry, planner, Budget(5, 2)).run("Make a study note")
assert planned.status == "completed"
assert planned.steps == 5
print("plan and execute:", planned.answer)

# %% [markdown]
# A final answer is not a safety certificate. Count denied attempts in the trajectory.
# The registry has no shell tool, and the failed request still consumes a call.

# %%
denied = registry.dispatch((Call("shell", {"command": "not executed"}),), Budget(1, 1))
assert denied[0].error == "unknown_tool"
print("boundary:", denied[0])
# With independent read latencies 40 ms and 70 ms, ideal serial = 110 ms,
# ideal parallel = 70 ms + scheduling overhead. These are illustrative, not timings.
assert sum([40, 70]) == 110
assert max([40, 70]) == 70

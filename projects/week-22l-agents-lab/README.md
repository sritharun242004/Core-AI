# Track L alpha — bounded agents lab

A runnable, dependency-free Python 3.13 laboratory for **ReAct**, **Reflexion-style
retry**, and **Plan-and-Execute** control flow. Scripted model adapters make every
choice observable and reproducible. They are not language models, paper
reproductions, or evidence of real-world autonomous task success.

## Run offline

From the repository root (the repository's existing `.venv` must have pytest and Ruff):

```bash
PYTHONPATH=projects/week-22l-agents-lab/src .venv/bin/python -m pytest projects/week-22l-agents-lab/tests
PYTHONPATH=projects/week-22l-agents-lab/src .venv/bin/python projects/week-22l-agents-lab/notebooks/01_bounded_agents.py
.venv/bin/ruff check projects/week-22l-agents-lab
```

The percent-format notebook is ordinary executable Python. Jupytext can convert it
to `.ipynb` in a separately managed environment; conversion is not required.
No model, dataset, network, credential, GPU, or cloud account is used.

## Public API

- `Tool(name, parameters, output, handler, read_only=True)` declares a trusted
  local callback. Exact argument keys and primitive types are checked before use;
  strings are at most 4,096 characters and numeric magnitudes at most 1,000.
  Booleans are not numeric arguments; NaN/infinity and additional keys are rejected.
  Results obey the same primitive bounds. This is a **small typed schema**, not
  a general JSON Schema implementation. Register tools at the host boundary only.
- `ToolRegistry(max_tools=16, max_batch=8)` is an allowlist. The fixture exposes
  only bounded addition and two public in-memory records. Unknown/private keys
  produce stable errors, never an exception traceback with secret data.
- `Call`, `Action`, `Context`, `Observation`, `Event`, `RunResult` are explicit
  structured messages. An action is either tool calls or a final answer, never both.
- `Budget(max_steps, max_calls)` charges each model/planning attempt and each
  admitted tool attempt, including invalid/denied calls. Reserve a whole batch
  before dispatch; an over-budget batch executes nothing and charges nothing.
- `dispatch(..., parallel=True)` runs independent **read-only** callbacks on up to
  four threads and joins in input order. No dependency substitution exists;
  dependent calls require separate model steps. The barrier test checks actual
  concurrent dispatch, not just a fabricated latency estimate.
- `BoundedMemory` retains FIFO feedback/subtask summaries. `ScriptedModel` records
  copied contexts; `ReAct.run` executes the observe/act loop, `Reflexion.run` retries
  with trusted verifier feedback, and `PlanAndExecute.run` validates a scripted
  plan before executing its subtasks under one shared budget.

```python
from agents_lab import Action, Budget, Call, ReAct, ScriptedModel, fixture_registry

model = ScriptedModel(
    [
        Action(calls=(Call("add", {"a": 2, "b": 3}),)),
        Action(final="5"),
    ]
)
result = ReAct(fixture_registry(), model, Budget(3, 1)).run("2+3")
assert result.answer == "5"
```

`completed` means the adapter supplied a final answer, **not** that it is correct
or safe. Reflexion adds a trusted success predicate. Other statuses distinguish
budget limits, invalid actions/plans, model exhaustion/errors, verifier errors,
and failed verification. Events record decisions and observations, not private
chain-of-thought. Fresh controller, model, memory and budget instances isolate tasks.

## Architecture and limits

```
trusted host budget + registry
             ↓
model adapter → typed action → schema/allowlist → bounded local tool
      ↑                                            ↓
      └──────────── copied observation ─────────────┘
```

Memory and retrieved text are data, not authorization. A multi-agent system can
use several adapters, but delegation must carry the *same* global budget and
principal; giving each agent a fresh budget silently amplifies permissions/cost.
This lab does not implement a distributed multi-agent framework or durable memory.

Callbacks are trusted Python, **not sandboxed**. Threads cannot safely kill a hung
callback. Step/call/output bounds are not wall-clock limits; production execution
needs process isolation, deadlines, cancellation, per-user authorization, approval
for consequential actions, privacy-aware audit logs, idempotency and recovery.
No arbitrary shell, filesystem, or network capability is installed by this project.

## Learning path

Read `assignments/warmup.md`, `build.md`, and `challenge.md`; compare against
`SOLUTION_NOTES.md`. `COMPUTE.md` explains why the curriculum badge is yellow even
though the reference suite is cheap local CPU work. Continue to Track L beta for
MCP/A2A transport contracts, evaluation and token economics.

Primary sources: [ReAct](https://arxiv.org/abs/2210.03629),
[Reflexion](https://arxiv.org/abs/2303.11366),
[Plan-and-Solve](https://arxiv.org/abs/2305.04091). Plan-and-Solve is related
planning literature, not a claim that this generic Plan-and-Execute controller
reproduces that paper's prompting or results.

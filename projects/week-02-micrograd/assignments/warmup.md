# Warmup (30 min)

1. Add a `Value.tanh()` op with the correct backward rule. Add a test that verifies `tanh(0).backward()` leaves grad 1.0.
2. Add a `sigmoid()` op derived from `exp`. No new `_backward` needed — see how `__truediv__` was expressed.
3. Print the topological order of a small graph before running `backward()`. Confirm each parent appears before every child in the reversed list.

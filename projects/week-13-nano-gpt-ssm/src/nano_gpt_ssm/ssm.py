"""An intentionally explicit state-space/recurrent language-model baseline.

This is a teaching baseline inspired by the broad idea of input-dependent
state updates. It is not an implementation of a production Mamba, Griffin, or
other named architecture.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn


def diagonal_ssm_scan(
    inputs: Tensor,
    *,
    decay: Tensor,
    write: Tensor,
    read: Tensor,
    skip: Tensor,
    initial_state: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Scan independent state channels with input-dependent diagonal dynamics."""

    if inputs.ndim != 3 or decay.shape != inputs.shape or write.shape != inputs.shape:
        raise ValueError("inputs, decay, and write must have the same (batch, time, state) shape")
    if inputs.shape[1] == 0:
        raise ValueError("inputs must contain at least one time step")
    if read.shape != inputs.shape or skip.shape != (inputs.shape[-1],):
        raise ValueError("read and skip do not match the state width")
    if initial_state is None:
        state = inputs.new_zeros(inputs.shape[0], inputs.shape[-1])
    else:
        if initial_state.shape != (inputs.shape[0], inputs.shape[-1]):
            raise ValueError("initial_state must have shape (batch, state_size)")
        state = initial_state
    outputs: list[Tensor] = []
    for input_t, decay_t, write_t, read_t in zip(
        inputs.unbind(1), decay.unbind(1), write.unbind(1), read.unbind(1), strict=True
    ):
        state = decay_t * state + write_t * input_t
        outputs.append(read_t * state + skip * input_t)
    return torch.stack(outputs, dim=1), state


def ssm_recurrence(
    inputs: Tensor,
    transition: Tensor,
    input_projection: Tensor,
    *,
    initial_state: Tensor | None = None,
    activation: Callable[[Tensor], Tensor] | None = torch.tanh,
) -> Tensor:
    """Apply the explicit recurrence ``s_t = f(A s_(t-1) + B x_t)``.

    ``inputs`` is ``(batch, time, input_size)``, ``transition`` is ``(state,
    state)``, and ``input_projection`` is ``(state, input_size)``. Keeping the
    loop visible makes the state update a useful comparison against attention.
    """

    if inputs.ndim != 3 or transition.ndim != 2 or input_projection.ndim != 2:
        raise ValueError("inputs, transition, and input_projection have invalid ranks")
    if inputs.shape[1] == 0:
        raise ValueError("inputs must contain at least one time step")
    state_size, input_size = input_projection.shape
    if transition.shape != (state_size, state_size) or inputs.shape[-1] != input_size:
        raise ValueError("recurrence matrices do not match the input dimensions")
    if initial_state is None:
        state = inputs.new_zeros(inputs.shape[0], state_size)
    else:
        if initial_state.shape != (inputs.shape[0], state_size):
            raise ValueError("initial_state must have shape (batch, state_size)")
        state = initial_state
    states: list[Tensor] = []
    for input_t in inputs.unbind(dim=1):
        state = state @ transition.transpose(0, 1) + input_t @ input_projection.transpose(0, 1)
        if activation is not None:
            state = activation(state)
        states.append(state)
    return torch.stack(states, dim=1)


class SelectiveSSM(nn.Module):
    """A compact input-dependent diagonal SSM inspired by selective scans.

    The recurrence is deliberately explicit and educational; it is not a
    claim to reproduce any private or optimized Mamba implementation.
    """

    def __init__(self, state_size: int) -> None:
        super().__init__()
        if state_size <= 0:
            raise ValueError("state_size must be positive")
        self.state_size = state_size
        self.decay_projection = nn.Linear(state_size, state_size)
        self.write_projection = nn.Linear(state_size, state_size)
        self.read_projection = nn.Linear(state_size, state_size)
        self.skip = nn.Parameter(torch.zeros(state_size))

    def coefficients(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if inputs.ndim != 3 or inputs.shape[-1] != self.state_size or inputs.shape[1] == 0:
            raise ValueError("inputs must have shape (batch, nonempty time, state_size)")
        decay = torch.sigmoid(self.decay_projection(inputs))
        write = torch.tanh(self.write_projection(inputs))
        read = torch.sigmoid(self.read_projection(inputs))
        return decay, write, read

    def forward(self, inputs: Tensor, initial_state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if inputs.ndim != 3 or inputs.shape[-1] != self.state_size or inputs.shape[1] == 0:
            raise ValueError("inputs must have shape (batch, nonempty time, state_size)")
        decay, write, read = self.coefficients(inputs)
        return diagonal_ssm_scan(
            inputs,
            decay=decay,
            write=write,
            read=read,
            skip=self.skip,
            initial_state=initial_state,
        )


class ExplicitSSMCell(nn.Module):
    """A transparent recurrent state update with an optional input gate."""

    def __init__(self, input_size: int, state_size: int, *, gated: bool = True) -> None:
        super().__init__()
        if input_size <= 0 or state_size <= 0:
            raise ValueError("input_size and state_size must be positive")
        self.input_size = input_size
        self.state_size = state_size
        self.gated = gated
        self.input_projection = nn.Linear(input_size, state_size)
        self.state_projection = nn.Linear(state_size, state_size, bias=False)
        self.gate_projection = nn.Linear(input_size, state_size) if gated else None

    def step(self, input_t: Tensor, state: Tensor) -> Tensor:
        if input_t.ndim != 2 or input_t.shape[-1] != self.input_size:
            raise ValueError("input_t must have shape (batch, input_size)")
        if state.shape != (input_t.shape[0], self.state_size):
            raise ValueError("state must have shape (batch, state_size) matching input_t")
        candidate = torch.tanh(self.input_projection(input_t) + self.state_projection(state))
        if self.gate_projection is None:
            return candidate
        gate = torch.sigmoid(self.gate_projection(input_t))
        return gate * candidate + (1 - gate) * state

    def forward(
        self,
        inputs: Tensor,
        initial_state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_size or inputs.shape[1] == 0:
            raise ValueError("inputs must have shape (batch, nonempty time, input_size)")
        if initial_state is None:
            state = inputs.new_zeros(inputs.shape[0], self.state_size)
        else:
            if initial_state.shape != (inputs.shape[0], self.state_size):
                raise ValueError("initial_state must have shape (batch, state_size)")
            state = initial_state
        states: list[Tensor] = []
        for input_t in inputs.unbind(dim=1):
            state = self.step(input_t, state)
            states.append(state)
        return torch.stack(states, dim=1), state


class TinySSMLanguageModel(nn.Module):
    """A tiny gated state-space language model for a matched toy comparison."""

    def __init__(
        self,
        vocab_size: int,
        *,
        embedding_size: int = 32,
        state_size: int = 48,
        gated: bool = True,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, embedding_size)
        self.recurrence = ExplicitSSMCell(embedding_size, state_size, gated=gated)
        self.lm_head = nn.Linear(state_size, vocab_size)

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2 or input_ids.dtype != torch.long or input_ids.shape[1] == 0:
            raise ValueError("input_ids must be a rank-2 torch.long tensor with nonempty time")
        if input_ids.numel() and (
            int(input_ids.min()) < 0 or int(input_ids.max()) >= self.vocab_size
        ):
            raise ValueError("input_ids contains an id outside the vocabulary")
        states, _ = self.recurrence(self.token_embedding(input_ids))
        return self.lm_head(states)


TinySSM = TinySSMLanguageModel
MambaStyleBaseline = TinySSMLanguageModel

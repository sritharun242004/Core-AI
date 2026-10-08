# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
"""Narrow signatures for unannotated PyTorch seed/autograd entry points."""

from collections.abc import Callable
from typing import cast

import torch
from torch import Tensor

# These casts describe only the call forms used in this lab; they do not
# claim to type PyTorch's other optional backward arguments.
manual_seed = cast(Callable[[int], torch.Generator], torch.manual_seed)
backward = cast(Callable[[Tensor], None], Tensor.backward)

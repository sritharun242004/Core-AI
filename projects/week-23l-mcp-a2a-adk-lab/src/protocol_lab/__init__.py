"""Offline protocol, evaluation and economics teaching laboratory."""

from .a2a import A2AClient, A2AServer
from .economics import (
    CacheKey,
    ModelOption,
    Price,
    ResponseCache,
    TokenUsage,
    estimate_cost,
    route_model,
)
from .evaluation import (
    CaseResult,
    EvaluationAttempt,
    EvaluationReport,
    RepositoryTask,
    apply_replacements,
    evaluate,
    repository_fixtures,
)
from .mcp import MCPClient, MCPServer
from .rpc import LocalTransport, ProtocolError, serve_lines

__all__ = [
    "A2AClient",
    "A2AServer",
    "CacheKey",
    "CaseResult",
    "EvaluationAttempt",
    "EvaluationReport",
    "LocalTransport",
    "MCPClient",
    "MCPServer",
    "ModelOption",
    "Price",
    "ProtocolError",
    "RepositoryTask",
    "ResponseCache",
    "TokenUsage",
    "apply_replacements",
    "estimate_cost",
    "evaluate",
    "repository_fixtures",
    "route_model",
    "serve_lines",
]

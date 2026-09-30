"""Run the local MCP teaching server over newline-delimited stdin/stdout."""

import sys

from .mcp import MCPServer
from .rpc import serve_lines


def main() -> None:
    serve_lines(MCPServer(), sys.stdin, sys.stdout)


if __name__ == "__main__":
    main()

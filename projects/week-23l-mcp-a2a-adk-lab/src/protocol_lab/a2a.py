"""Explicit A2A-inspired local task teaching subset v0.1, NOT wire-compliant A2A.

A2A and ADK are Google Cloud initiatives, not Google DeepMind products.
Method names and task envelopes here are reduced for lifecycle exercises.
"""

from copy import deepcopy

from .rpc import JSONRPCServer, ProtocolError, RPCClient

SCOPE = "local-a2a-teaching-subset/0.1; no A2A conformance claim"
TERMINAL = {"completed", "failed", "canceled"}
STATES = {"submitted", "working"} | TERMINAL


class A2AServer(JSONRPCServer):
    def __init__(self, max_tasks: int = 32):
        if type(max_tasks) is not int or not 1 <= max_tasks <= 1000:
            raise ValueError("invalid task capacity")
        self.max_tasks = max_tasks
        self._tasks: dict[str, dict] = {}

    def dispatch(self, method: str, params: dict, *, notification: bool):
        if notification:
            return None
        if method == "agent/card":
            return {
                "name": "Public guide assistant",
                "scope": SCOPE,
                "skills": ["summarize-public-guide"],
                "authentication": "none-local-only",
            }
        if method not in {"tasks/send", "tasks/get", "tasks/cancel", "demo/advance"}:
            raise ProtocolError(-32601, "method not found")
        expected = {"id", "message"} if method == "tasks/send" else {"id"}
        if (
            params.keys() != expected
            or not isinstance(params.get("id"), str)
            or (not 1 <= len(params["id"]) <= 64)
        ):
            raise ProtocolError(-32602, "invalid task envelope")
        ident = params["id"]
        if method == "tasks/send":
            message = params["message"]
            if not isinstance(message, str) or not 1 <= len(message) <= 1024:
                raise ProtocolError(-32602, "invalid message")
            if ident in self._tasks:
                if self._tasks[ident]["message"] != message:
                    raise ProtocolError(-32010, "task id conflicts with existing request")
                return deepcopy(self._tasks[ident])
            if len(self._tasks) >= self.max_tasks:
                raise ProtocolError(-32011, "task capacity exceeded")
            self._tasks[ident] = {
                "id": ident,
                "message": message,
                "status": {"state": "submitted"},
                "artifacts": [],
            }
            return deepcopy(self._tasks[ident])
        if ident not in self._tasks:
            raise ProtocolError(-32004, "task not found")
        task = self._tasks[ident]
        state = task["status"]["state"]
        if method == "tasks/get":
            return deepcopy(task)
        if state in TERMINAL:
            raise ProtocolError(-32012, "terminal task is immutable")
        if method == "tasks/cancel":
            task["status"] = {"state": "canceled"}
        elif state == "submitted":
            task["status"] = {"state": "working"}
        elif "guide" in task["message"].lower():
            task["status"] = {"state": "completed"}
            task["artifacts"] = [{"text": "Use public fixtures only."}]
        else:
            task["status"] = {"state": "failed", "message": "only public guide tasks are supported"}
        return deepcopy(task)


class A2AClient(RPCClient):
    def _task(self, method: str, params: dict) -> dict:
        task = self.request(method, params)
        if (
            not isinstance(task, dict)
            or task.get("id") != params["id"]
            or (
                not isinstance(task.get("status"), dict)
                or not isinstance(task["status"].get("state"), str)
                or task["status"]["state"] not in STATES
                or not isinstance(task.get("artifacts"), list)
            )
        ):
            raise ProtocolError(-32600, "invalid task result")
        return task

    def submit(self, task_id: str, message: str) -> dict:
        return self._task("tasks/send", {"id": task_id, "message": message})

    def get(self, task_id: str) -> dict:
        return self._task("tasks/get", {"id": task_id})

    def cancel(self, task_id: str) -> dict:
        return self._task("tasks/cancel", {"id": task_id})

    def advance(self, task_id: str) -> dict:
        """Test-only explicit scheduler tick, not an A2A standard method."""
        return self._task("demo/advance", {"id": task_id})

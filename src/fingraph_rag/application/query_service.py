"""Bounded request lifecycle with durable LangGraph checkpoints and replayable events."""

import asyncio
import time
from contextlib import AsyncExitStack
from pathlib import Path
from uuid import uuid4

from fingraph_rag.observability import get_telemetry
from fingraph_rag.workflow.build_graph import build_graph
from fingraph_rag.workflow.events import EventStore


class QueryService:
    def __init__(
        self,
        retrieval,
        generator,
        policies,
        *,
        state_directory: str,
        timeout_seconds: float = 30,
        max_active: int = 32,
        max_repairs: int = 2,
        dependency_probe=None,
    ):
        self.retrieval, self.generator, self.policies = retrieval, generator, policies
        self.state_directory = Path(state_directory)
        self.timeout_seconds, self.max_active = timeout_seconds, max_active
        if (
            not 0 < timeout_seconds <= 300
            or not 1 <= max_active <= 256
            or not 0 <= max_repairs <= 2
        ):
            raise ValueError("invalid request budgets")
        self.max_repairs = max_repairs
        self.dependency_probe = dependency_probe
        self.tasks = {}
        self.stack = AsyncExitStack()

    async def open(self):
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        await self.check_ready()
        self.state_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        events = EventStore(str(self.state_directory / "events.sqlite"))
        try:
            checkpointer = await self.stack.enter_async_context(
                AsyncSqliteSaver.from_conn_string(str(self.state_directory / "checkpoints.sqlite"))
            )
            await checkpointer.setup()
            graph = build_graph(
                self.retrieval,
                self.generator,
                self.policies,
                events,
                checkpointer,
                max_repairs=self.max_repairs,
            )
        except Exception:
            try:
                await self.stack.aclose()
            finally:
                events.close()
            raise
        self.events = events
        self.graph = graph
        return self

    async def check_ready(self):
        if self.dependency_probe:
            await asyncio.wait_for(self.dependency_probe(), timeout=10)

    async def close(self):
        for task in list(self.tasks.values()):
            task.cancel()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)
        try:
            await self.stack.aclose()
        finally:
            if hasattr(self, "events"):
                self.events.close()

    def submit(self, question):
        if len(self.tasks) >= self.max_active:
            raise RuntimeError("request_capacity_exceeded")
        request_id = uuid4().hex
        self.events.append(request_id, "accepted")
        task = asyncio.create_task(self._run(request_id, question))
        self.tasks[request_id] = task
        task.add_done_callback(lambda _: self.tasks.pop(request_id, None))
        return request_id

    async def _run(self, request_id, question):
        terminal = "error"
        route = "unknown"
        started = time.perf_counter()
        try:
            state = await asyncio.wait_for(
                self.graph.ainvoke(
                    {"request_id": request_id, "question": question},
                    config={"configurable": {"thread_id": request_id}, "recursion_limit": 24},
                ),
                timeout=self.timeout_seconds,
            )
            route = state.get("query_spec", {}).get("route", "unknown")
            terminal = state["terminal"]
        except TimeoutError:
            self.events.append(request_id, "error", {"code": "query_timeout"})
        except asyncio.CancelledError:
            self.events.append(request_id, "error", {"code": "query_cancelled"})
            raise
        except Exception:
            self.events.append(request_id, "error", {"code": "query_failed"})
        finally:
            self.events.append(request_id, "done", {"outcome": terminal})
            outcome = "abstained" if terminal == "abstain" else terminal
            get_telemetry().metrics.record_request(
                route, outcome, (time.perf_counter() - started) * 1000
            )

    async def stream(self, request_id, after=-1):
        while True:
            rows = self.events.read(request_id, after)
            for event in rows:
                after = event.sequence
                yield event
                if event.event_type == "done":
                    return
            if request_id not in self.tasks:
                # A restarted worker cannot silently continue an interrupted request.
                if rows or self.events.exists(request_id):
                    prior = self.events.read(request_id)
                    if prior and prior[-1].event_type != "done":
                        self.events.append(request_id, "error", {"code": "request_interrupted"})
                        self.events.append(request_id, "done", {"outcome": "error"})
                        continue
                return
            await asyncio.sleep(0.02)

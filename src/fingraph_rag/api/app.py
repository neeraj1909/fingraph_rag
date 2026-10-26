"""FastAPI SSE with verification-before-answer and durable event replay.

Run one worker per SQLite state directory. Disconnect stops streaming but leaves
bounded computation running; reconnect via GET with Last-Event-ID. Service
restart terminates interrupted requests with stable error/done on replay.
"""

import hmac
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from fingraph_rag.api.dependencies import configured_service
from fingraph_rag.api.schemas import QueryRequest


def create_app(service_factory=configured_service):
    @asynccontextmanager
    async def lifespan(app):
        from fingraph_rag.observability import configure_otel, set_telemetry

        telemetry = configure_otel(
            endpoint=os.getenv("FINGRAPH_OTLP_ENDPOINT"),
            output_path=Path(os.environ["FINGRAPH_TRACE_PATH"])
            if os.getenv("FINGRAPH_TRACE_PATH")
            else None,
        )
        set_telemetry(telemetry)
        app.state.service = None
        app.state.ready = False
        service = None
        try:
            service = service_factory()
            await service.open()
        except Exception:
            if service is not None:
                await service.close()
            # Readiness is explicit; never silently fall back to fixture answers.
            app.state.configuration_error = "runtime_unavailable"
        else:
            app.state.service = service
            app.state.ready = True
        yield
        if app.state.service:
            await app.state.service.close()
        telemetry.flush()
        telemetry.shutdown()

    app = FastAPI(title="FinGraph-RAG", lifespan=lifespan)

    @app.get("/health/live")
    async def live():
        return {"status": "alive"}

    @app.get("/health/ready")
    async def ready(request: Request):
        if not request.app.state.ready:
            raise HTTPException(503, "runtime_unavailable")
        try:
            await request.app.state.service.check_ready()
        except Exception:
            raise HTTPException(503, "dependency_unavailable") from None
        return {"status": "ready"}

    def service(request):
        token = os.getenv("FINGRAPH_API_TOKEN")
        if token:
            if not hmac.compare_digest(request.headers.get("authorization", ""), "Bearer " + token):
                raise HTTPException(
                    401, "authentication_required", headers={"WWW-Authenticate": "Bearer"}
                )
        elif request.client is None or request.client.host not in {
            "127.0.0.1",
            "::1",
            "localhost",
            "testclient",
        }:
            raise HTTPException(403, "local_development_only")
        if not request.app.state.ready:
            raise HTTPException(503, "runtime_unavailable")
        return request.app.state.service

    def stream(query_service, request_id, after=-1):
        async def encoded():
            async for event in query_service.stream(request_id, after):
                yield (
                    f"id: {event.sequence}\nevent: {event.event_type}\ndata: "
                    + event.model_dump_json()
                    + "\n\n"
                )

        return StreamingResponse(
            encoded(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Request-ID": request_id,
            },
        )

    @app.post("/v1/query")
    async def query(body: QueryRequest, request: Request):
        query_service = service(request)
        try:
            request_id = query_service.submit(body.question)
        except RuntimeError:
            raise HTTPException(429, "request_capacity_exceeded") from None
        return stream(query_service, request_id)

    @app.get("/v1/query/{request_id}/events")
    async def replay(request_id: str, request: Request, last_event_id: str | None = Header(None)):
        query_service = service(request)
        if not query_service.events.exists(request_id):
            raise HTTPException(404, "request_not_found")
        try:
            after = int(last_event_id) if last_event_id is not None else -1
            if after < -1:
                raise ValueError
        except ValueError:
            raise HTTPException(400, "invalid_event_cursor") from None
        return stream(query_service, request_id, after)

    return app


app = create_app()

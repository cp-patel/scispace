"""HTTP API. It only reads and writes rows; workers do the slow work.

Run with: uvicorn research_assistant.api.app:app
"""

import hmac
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from research_assistant.config import JobSettings, Settings, get_settings
from research_assistant.db import apply_schema, create_pool, jobs, results
from research_assistant.observability import setup_logging

log = logging.getLogger(__name__)


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status, self.code, self.message = status, code, message


class ApproveIn(BaseModel):
    remove: list[str] = Field(default_factory=list, max_length=100)
    add: list[str] = Field(default_factory=list, max_length=20)


def error_body(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging(settings.log_level)
        if settings.api_key == "change-me":
            log.warning("API_KEY is the default value; set a long random API_KEY in .env before exposing this server")
        await apply_schema(settings.database_url)
        app.state.pool = await create_pool(settings.database_url)
        yield
        await app.state.pool.close()

    app = FastAPI(title="Research Assistant", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(ApiError)
    async def api_error(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(error_body(exc.code, exc.message), status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def invalid(_: Request, exc: RequestValidationError) -> JSONResponse:
        problems = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
        return JSONResponse(error_body("invalid_input", problems), status_code=422)

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        if x_api_key is None or not hmac.compare_digest(x_api_key.encode(), settings.api_key.encode()):
            raise ApiError(401, "unauthorized", "missing or wrong X-API-Key header")

    def pool(request: Request):
        return request.app.state.pool

    async def existing_job(job_id: uuid.UUID, request: Request) -> dict:
        job = await jobs.get_job(request.app.state.pool, str(job_id))
        if job is None:
            raise ApiError(404, "not_found", "no job with that id")
        return job

    async def finished_output(job: dict, request: Request) -> dict:
        out = await results.outputs(request.app.state.pool, str(job["id"]))
        if out is None:
            raise ApiError(404, "not_ready", f"no output yet; the job is {job['status']}")
        return out

    auth = [Depends(require_key)]

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(request: Request) -> dict:
        async with request.app.state.pool.connection() as conn:
            await conn.execute("SELECT 1")
        return {"status": "ready"}

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/jobs", status_code=202, dependencies=auth)
    async def create(body: JobSettings, request: Request) -> dict:
        if (body.brain or settings.brain) == "claude" and not settings.anthropic_api_key:
            raise ApiError(422, "invalid_input", "the claude brain needs ANTHROPIC_API_KEY on the server")
        if body.budget_usd is None:
            body = body.model_copy(update={"budget_usd": settings.job_budget_usd})
        return {"job_id": await jobs.create_job(pool(request), body)}

    @app.get("/v1/jobs", dependencies=auth)
    async def list_all(request: Request, limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)) -> list[dict]:
        return await jobs.list_jobs(pool(request), limit, offset)

    @app.get("/v1/jobs/{job_id}", dependencies=auth)
    async def get_one(request: Request, job: dict = Depends(existing_job)) -> dict:
        events = await jobs.events(pool(request), str(job["id"]))
        cost = sum((await jobs.cost_by_node(pool(request), str(job["id"]))).values())
        return {
            "id": job["id"], "topic": job["topic"], "status": job["status"], "attempts": job["attempts"],
            "error": job["error"], "created_at": job["created_at"], "updated_at": job["updated_at"],
            "current_step": events[-1]["node"] if events else None,
            "progress": [{"node": e["node"], "event": e["event"], **e["data"]} for e in events if e["event"] == "end"],
            "cost_usd": round(cost, 6),
        }

    @app.get("/v1/jobs/{job_id}/selection", dependencies=auth)
    async def selection(request: Request, job: dict = Depends(existing_job)) -> list[dict]:
        return await results.selection_table(pool(request), str(job["id"]))

    @app.post("/v1/jobs/{job_id}/approve", dependencies=auth)
    async def approve(body: ApproveIn, request: Request, job: dict = Depends(existing_job)) -> dict:
        try:
            await jobs.approve(pool(request), str(job["id"]), body.model_dump())
        except jobs.StateConflict as exc:
            raise ApiError(409, "conflict", str(exc)) from exc
        return {"status": "queued"}

    @app.post("/v1/jobs/{job_id}/cancel", dependencies=auth)
    async def cancel(request: Request, job: dict = Depends(existing_job)) -> dict:
        try:
            await jobs.cancel(pool(request), str(job["id"]))
        except jobs.StateConflict as exc:
            raise ApiError(409, "conflict", str(exc)) from exc
        return {"status": "canceled"}

    @app.get("/v1/jobs/{job_id}/paper", dependencies=auth)
    async def paper(request: Request, job: dict = Depends(existing_job)) -> PlainTextResponse:
        return PlainTextResponse((await finished_output(job, request))["paper_md"], media_type="text/markdown")

    @app.get("/v1/jobs/{job_id}/references.bib", dependencies=auth)
    async def bib(request: Request, job: dict = Depends(existing_job)) -> PlainTextResponse:
        return PlainTextResponse((await finished_output(job, request))["references_bib"], media_type="application/x-bibtex")

    @app.get("/v1/jobs/{job_id}/report", dependencies=auth)
    async def report(request: Request, job: dict = Depends(existing_job)) -> dict:
        return (await finished_output(job, request))["report"]

    @app.get("/v1/jobs/{job_id}/trace", dependencies=auth)
    async def trace(request: Request, job: dict = Depends(existing_job)) -> list[dict]:
        cost = await jobs.cost_by_node(pool(request), str(job["id"]))
        rows = []
        for e in await jobs.events(pool(request), str(job["id"])):
            if e["event"] == "end":
                rows.append({"node": e["node"], "finished_at": e["created_at"], "cost_usd": cost.get(e["node"], 0.0), **e["data"]})
        return rows

    return app


app = create_app()

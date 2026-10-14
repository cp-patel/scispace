"""Logs, metrics, and the per-job context that logging and cost tracking both read.

The job context is a contextvar: set once when a step starts, it follows every await inside that
step, so a log line or an LLM bill can be tagged with the job and step without passing ids around.
"""

import contextvars
import json
import logging
from dataclasses import dataclass

from prometheus_client import Counter, Gauge, Histogram


@dataclass(frozen=True)
class JobContext:
    job_id: str | None = None
    node: str | None = None
    worker_id: str | None = None
    budget_usd: float | None = None


job_context: contextvars.ContextVar[JobContext] = contextvars.ContextVar("job_context", default=JobContext())  # noqa: B039 (frozen, so immutable)

JOBS_TOTAL = Counter("jobs_total", "Jobs that reached a final status", ["status"])
NODE_SECONDS = Histogram(
    "node_duration_seconds", "Time spent in each agent step", ["node"],
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600),
)
LLM_TOKENS = Counter("llm_tokens_total", "LLM tokens by kind (input, output, cache_read, cache_write)", ["model", "kind"])
LLM_COST = Counter("llm_cost_usd_total", "LLM spend in US dollars", ["model"])
EXTERNAL_REQUESTS = Counter("external_requests_total", "Calls to outside APIs", ["service", "outcome"])
QUEUE_DEPTH = Gauge("queue_depth", "Jobs waiting in the queue")


class JsonFormatter(logging.Formatter):
    EXTRA = ("attempt", "worker_id", "duration_ms", "status", "service")

    def format(self, record: logging.LogRecord) -> str:
        ctx = job_context.get()
        out = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "job_id": ctx.job_id,
            "node": ctx.node,
        }
        for key in self.EXTRA:
            if hasattr(record, key):
                out[key] = getattr(record, key)
        if record.exc_info:
            out["error"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)

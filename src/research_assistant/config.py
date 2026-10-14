"""All settings in one place, read from environment variables or a .env file.

Algorithm defaults live in AlgoParams and can be overridden with a double underscore,
for example ALGO__RRF_K=30.
"""

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Effort = Literal["low", "medium", "high", "xhigh", "max"]
SelectionMethod = Literal["submodular", "dpp", "mmr", "topk"]
STEPS = ("plan", "screen", "name_sections", "compress", "write_section", "check_support", "revise", "write_frame")


class AlgoParams(BaseModel):
    # keyword and vector scoring
    bm25_k1: float = 1.2
    bm25_title_weight: float = 2.0
    bm25_title_b: float = 0.3
    bm25_abstract_weight: float = 1.0
    bm25_abstract_b: float = 0.75
    rocchio_alpha: float = 1.0
    rocchio_beta: float = 0.75
    rocchio_gamma: float = 0.0
    rocchio_top: int = 20
    # citation graph
    ppr_alpha: float = 0.85
    ppr_tol: float = 1e-10
    ppr_max_iter: int = 200
    ppr_seeds: int = 50
    # fusion
    rrf_k: int = 60
    rrf_graph_weight: float = 1.0
    # gathering the pool
    pool_cap: int = 3000
    local_top: int = 200
    openalex_pages: int = 2
    snowball_seeds: int = 30
    snowball_forward_seeds: int = 10
    # screening and selection
    screen_batch: int = 10
    screen_concurrency: int = 4
    prior_top: float = 0.6
    prior_bottom: float = 0.1
    select_mu: float = 1.0
    dpp_theta: float = 2.0
    dpp_eps: float = 1e-6
    mmr_lambda: float = 0.7
    # ingest and writing
    chunk_tokens: int = 500
    chunk_overlap: int = 50
    pdf_max_bytes: int = 20_000_000
    min_fulltext_chars: int = 1000
    cluster_beta: float = 0.7
    cluster_knn: int = 8
    section_chunks: int = 40
    section_budget_tokens: int = 6000
    max_cards: int = 8
    revise_rounds: int = 2
    min_section_sentences: int = 2
    uncited_max_words: int = 12
    support_threshold: float = 0.6
    partial_threshold: float = 0.3


class JobSettings(BaseModel):
    """What a client may set per job. Validated at the API boundary."""

    topic: Annotated[str, StringConstraints(strip_whitespace=True, min_length=10, max_length=300)]
    year_from: int | None = Field(default=None, ge=1900, le=2100)
    year_to: int | None = Field(default=None, ge=1900, le=2100)
    max_papers: int = Field(default=40, ge=10, le=100)
    shortlist_size: int = Field(default=150, ge=10, le=300)
    selection_method: SelectionMethod = "submodular"
    must_include: list[str] = Field(default_factory=list, max_length=20)
    auto_approve: bool = True
    budget_usd: float | None = Field(default=None, gt=0, le=100)  # None means the server default (JOB_BUDGET_USD)
    brain: Literal["offline", "claude"] | None = None
    published_before: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    exclude_ids: list[str] = Field(default_factory=list, max_length=50)

    @model_validator(mode="after")
    def _check(self) -> "JobSettings":
        if self.shortlist_size < self.max_papers:
            raise ValueError("shortlist_size must be at least max_papers")
        if self.year_from and self.year_to and self.year_from > self.year_to:
            raise ValueError("year_from must not be after year_to")
        return self


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter="__", extra="ignore")

    database_url: str = "postgresql://ra:ra@localhost:55432/ra"
    api_key: str = "change-me"
    brain: Literal["offline", "claude"] = "offline"
    anthropic_api_key: str | None = None
    openalex_api_key: str | None = None
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_cache_dir: str | None = None
    http_cache_dir: str | None = None
    calibration_path: str | None = "evals/results/calibration.json"
    log_level: str = "INFO"
    worker_metrics_port: int = 9100

    model_default: str = "claude-opus-5"
    model_plan: str | None = None
    model_screen: str | None = None
    model_name_sections: str | None = None
    model_compress: str | None = None
    model_write_section: str | None = None
    model_check_support: str | None = None
    model_revise: str | None = None
    model_write_frame: str | None = None
    effort_plan: Effort = "medium"
    effort_screen: Effort = "low"
    effort_name_sections: Effort = "medium"
    effort_compress: Effort = "medium"
    effort_write_section: Effort = "high"
    effort_check_support: Effort = "low"
    effort_revise: Effort = "high"
    effort_write_frame: Effort = "high"

    job_budget_usd: float = 10.0
    job_deadline_min: int = 30
    lease_seconds: int = 120
    heartbeat_seconds: float = 30.0
    max_attempts: int = 3

    algo: AlgoParams = Field(default_factory=AlgoParams)

    def model_for(self, step: str) -> str:
        return getattr(self, f"model_{step}") or self.model_default

    def effort_for(self, step: str) -> Effort:
        return getattr(self, f"effort_{step}")


@lru_cache
def get_settings() -> Settings:
    return Settings()

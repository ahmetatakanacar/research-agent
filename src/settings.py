import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent

INTERVIEW_TURNS = 3
SEARCH_DEPTH = "advanced"
SEARCH_RESULTS = 5
SEARCH_CANDIDATES = 10


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _float(name: str) -> float | None:
    try:
        return float(os.getenv(name, "").strip())
    except ValueError:
        return None


def _flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    openai_model: str
    fast_model: str
    llm_timeout: int
    llm_max_retries: int
    price_input_per_1m: float | None
    price_output_per_1m: float | None
    data_dir: Path
    api_key: str
    rate_limit_per_hour: int
    max_concurrent_runs: int
    trust_proxy: bool
    run_ttl_seconds: int
    log_level: str

    @property
    def app_db(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def checkpoint_db(self) -> Path:
        return self.data_dir / "checkpoints.db"


def get_settings() -> Settings:
    model = os.getenv("OPENAI_MODEL", "").strip() or "gpt-6-luna"
    data_dir = Path(os.getenv("DATA_DIR", "").strip() or ROOT_DIR / "data")
    return Settings(
        openai_model=model,
        fast_model=os.getenv("OPENAI_FAST_MODEL", "").strip() or model,
        llm_timeout=_int("LLM_TIMEOUT_SECONDS", 90),
        llm_max_retries=_int("LLM_MAX_RETRIES", 3),
        price_input_per_1m=_float("LLM_PRICE_INPUT_PER_1M"),
        price_output_per_1m=_float("LLM_PRICE_OUTPUT_PER_1M"),
        data_dir=data_dir,
        api_key=os.getenv("APP_API_KEY", "").strip(),
        rate_limit_per_hour=_int("RATE_LIMIT_PER_HOUR", 20),
        max_concurrent_runs=max(1, _int("MAX_CONCURRENT_RUNS", 3)),
        trust_proxy=_flag("TRUST_PROXY"),
        run_ttl_seconds=_int("RUN_TTL_SECONDS", 600),
        log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper() or "INFO",
    )

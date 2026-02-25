"""Configuration loaded from environment variables with sensible defaults."""

import json
import os
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class MeridianConfig:
    # Core
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("MERIDIAN_DATA_DIR", str(Path.home() / ".meridian"))))
    project_id: str = field(default_factory=lambda: os.getenv("MERIDIAN_PROJECT_ID", "default"))

    # Anthropic
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))

    # Qdrant
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333"))
    qdrant_collection: str = field(default_factory=lambda: os.getenv("QDRANT_COLLECTION", "memories"))

    # Ollama
    ollama_url: str = field(default_factory=lambda: os.getenv("OLLAMA_URL", "http://localhost:11434"))
    gateway_model: str = field(default_factory=lambda: os.getenv("GATEWAY_MODEL", "qwen2.5-coder:14b"))
    worker_model: str = field(default_factory=lambda: os.getenv("WORKER_MODEL", "qwen2.5-coder:14b"))
    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", "nomic-embed-text"))

    # Web Server
    host: str = field(default_factory=lambda: os.getenv("MERIDIAN_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("MERIDIAN_PORT", "7891")))
    working_dir: Path = field(default_factory=lambda: Path(os.getenv("MERIDIAN_WORKING_DIR", str(Path.home()))))

    # Gmail
    gmail_address: str = field(default_factory=lambda: os.getenv("GMAIL_ADDRESS", ""))
    gmail_app_password: str = field(default_factory=lambda: os.getenv("GMAIL_APP_PASSWORD", ""))
    gmail_token_path: Path = field(default_factory=lambda: Path(os.getenv("GMAIL_TOKEN_PATH", str(Path.home() / ".gmail_token.json"))))

    # Tuning
    hot_memory_max_tokens: int = field(default_factory=lambda: int(os.getenv("HOT_MEMORY_MAX_TOKENS", "2000")))
    recall_default_max_tokens: int = field(default_factory=lambda: int(os.getenv("RECALL_DEFAULT_MAX_TOKENS", "800")))
    gateway_synthesis_max_tokens: int = field(default_factory=lambda: int(os.getenv("GATEWAY_SYNTHESIS_MAX_TOKENS", "1000")))
    episodic_retention_days: int = field(default_factory=lambda: int(os.getenv("EPISODIC_RETENTION_DAYS", "90")))
    compact_threshold_percent: int = field(default_factory=lambda: int(os.getenv("COMPACT_THRESHOLD_PERCENT", "70")))

    def __post_init__(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "episodes").mkdir(exist_ok=True)
        self._load_accounts()

    def _load_accounts(self):
        """Load credentials from .accounts file if env vars aren't set."""
        accounts_path = Path(os.getenv("ACCOUNTS_FILE", str(Path.home() / ".accounts")))
        if not accounts_path.exists():
            return
        try:
            data = json.loads(accounts_path.read_text())
            email_info = data.get("email", {})
            if not self.gmail_address and email_info.get("address"):
                self.gmail_address = email_info["address"]
            if not self.gmail_app_password and email_info.get("password"):
                self.gmail_app_password = email_info["password"]
        except (json.JSONDecodeError, KeyError):
            pass


config = MeridianConfig()

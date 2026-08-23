"""Resolved settings: flags beat environment beats file beats defaults.

One resolution order, applied once, with `Config.sources` recording where each
value came from. `nl2sh config` prints that, because "why is it using that
model" is the question a layered config always eventually raises.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(os.environ.get("NL2SH_CONFIG",
                                  Path.home() / ".config" / "nl2sh" / "config.toml"))
CACHE_DIR = Path(os.environ.get("NL2SH_CACHE", Path.home() / ".cache" / "nl2sh"))

# `instantiate_anchored` is the default because it measured best on every
# column at 1B — routing 0.848 against generate's 0.799, literal reproduction
# 0.688 against 0.542, exact 0.280 against 0.195, on the same 164 rows.
#
# It is the WRONG default below about 1B. At 270M the same prompt collapses to
# 0.146 routing because the model answers in the shape of the source lines it
# was shown on 77% of rows, and `generate_anchored` wins there at 0.500. The
# CLI warns when a model that looks tiny is paired with this prompt; it does not
# silently switch, because the size of a model behind an HTTP endpoint is not
# something this program can know.
DEFAULT_PROMPT = "instantiate_anchored"
TINY_PROMPT = "generate_anchored"

ENV = {
    "backend": "NL2SH_BACKEND",
    "model": "NL2SH_MODEL",
    "base_url": "NL2SH_BASE_URL",
    "api_key": "NL2SH_API_KEY",
    "prompt": "NL2SH_PROMPT",
}


@dataclass
class Config:
    backend: str = "none"
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    prompt: str = DEFAULT_PROMPT
    k: int = 3
    max_tokens: int = 64
    temperature: float = 0.0
    sources: dict[str, str] = field(default_factory=dict)

    def backend_opts(self) -> dict[str, Any]:
        """Only the keys a backend constructor takes, and only if set."""
        opts: dict[str, Any] = {"model": self.model}
        if self.base_url:
            opts["base_url"] = self.base_url
        if self.api_key:
            opts["api_key"] = self.api_key
        return opts

    def redacted(self) -> dict:
        d = asdict(self)
        if d.get("api_key"):
            d["api_key"] = f"set ({len(self.api_key)} chars)"
        return d


def _load_file(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        import tomllib
    except ImportError:                                     # pragma: no cover
        return {}
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except Exception:                                       # noqa: BLE001
        # A malformed config must not stop the search path from working. The
        # CLI surfaces this as a warning; it is not worth a traceback.
        return {"__error__": f"could not parse {path}"}
    return data.get("nl2sh", data)


def resolve(**flags: Any) -> tuple[Config, list[str]]:
    """Build a Config, and return any warnings worth printing."""
    warnings: list[str] = []
    cfg = Config()
    from_file = _load_file(CONFIG_PATH)
    if "__error__" in from_file:
        warnings.append(from_file.pop("__error__"))

    for key in ("backend", "model", "base_url", "api_key", "prompt",
                "k", "max_tokens", "temperature"):
        value, origin = getattr(cfg, key), "default"
        if key in from_file and from_file[key] is not None:
            value, origin = from_file[key], str(CONFIG_PATH)
        env_name = ENV.get(key)
        if env_name and os.environ.get(env_name):
            value, origin = os.environ[env_name], f"${env_name}"
        if flags.get(key) is not None:
            value, origin = flags[key], "flag"
        if key in ("k", "max_tokens"):
            value = int(value)
        if key == "temperature":
            value = float(value)
        setattr(cfg, key, value)
        cfg.sources[key] = origin

    # A model named without a backend is a common slip, and defaulting to
    # search-only there would look like the model was ignored.
    if cfg.model and cfg.backend == "none":
        warnings.append(
            f"a model is set ({cfg.model!r}) but backend is 'none', so nothing "
            "will generate — pass --backend, or `nl2sh doctor` to see what is available")
    return cfg, warnings


def looks_tiny(model: str) -> bool:
    """A crude size sniff, used only to warn about the prompt default."""
    m = model.lower()
    return any(t in m for t in ("270m", "135m", "160m", "350m", "0.5b", "500m"))

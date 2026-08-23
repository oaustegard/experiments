"""Generator backends, all optional, none imported until asked for.

The product works with no backend at all — that path returns ranked
documentation and never reaches this module. Everything here is the *optional*
half, so the rules are:

* **No import at module load.** `llama_cpp`, `mlx_lm` and the HTTP clients are
  imported inside the method that needs them. A machine with none of them
  installed still runs `nl2sh search`.
* **stdlib HTTP only.** `urllib.request`, not `requests`, so a remote backend
  costs no dependency either.
* **A probe never raises and never returns a bare False.** It returns
  `Availability(ok, detail)` and the detail says *why* — "connection refused on
  http://localhost:11434" is actionable, `False` is not. A probe that degrades
  to a value a caller will act on is the failure mode this codebase has already
  paid for once.

Adding a backend means subclassing `Backend`, implementing `probe` and
`_complete`, and adding one line to `REGISTRY`.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

DEFAULT_MAX_TOKENS = 64          # stage 1's budget; see `Backend.generate`
DEFAULT_TIMEOUT = 120.0


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------

@dataclass
class Availability:
    """Whether a backend can run, and if not, what to do about it."""
    ok: bool
    detail: str = ""

    def __bool__(self) -> bool:
        return self.ok


@dataclass
class Generation:
    """One command, plus everything needed to judge whether to trust it."""
    command: str
    raw: str
    backend: str
    model: str
    seconds: float
    new_tokens: int | None = None
    truncated: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# parsing — the measured parser, not a new one
# --------------------------------------------------------------------------

THINK_CLOSE = "</think>"
_HEDGE = ("here", "sure", "to ", "you ", "this ", "okay", "certainly")


def strip_reasoning(gen: str) -> str:
    """Drop a reasoning trace so the parser reads the answer, not the thinking.

    Measured on Nemotron 3 Nano: under a 64-token budget the trace *is* the
    whole budget, and a parser that takes the first non-hedge line scores the
    first line of the model's scratchpad. Text after a closing `</think>` is the
    answer. An unterminated trace means the budget ran out before the model
    answered at all, and the empty string is the honest reading of that — the
    last line of someone's reasoning is not a command.
    """
    if THINK_CLOSE in gen:
        return gen.split(THINK_CLOSE)[-1]
    if "<think>" in gen:
        return ""
    return gen


def extract_command(gen: str) -> str:
    """A fenced or a bare command — `nl2sh-instantiate/run_gen.py`, unchanged.

    Kept byte-compatible with the harness that produced every number in
    `nl2sh-instantiate/RESULTS.md`, so a command this product prints is a
    command that eval scored. Changing it here silently invalidates that.
    """
    body = gen
    m = re.search(r"```(?:bash|sh|shell)?\s*\n?(.+?)```", gen, re.S)
    if m:
        body = m.group(1)
    for line in body.strip().splitlines():
        line = line.strip().strip("`").strip()
        if line and not line.lower().startswith(_HEDGE):
            return line
    return ""


def parse(gen: str) -> str:
    return extract_command(strip_reasoning(gen))


# --------------------------------------------------------------------------
# HTTP helper
# --------------------------------------------------------------------------

def _post_json(url: str, payload: dict, headers: dict | None = None,
               timeout: float = DEFAULT_TIMEOUT) -> dict:
    data = json.dumps(payload).encode()
    hdrs = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _get_json(url: str, headers: dict | None = None, timeout: float = 5.0) -> dict:
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _reach(url: str, headers: dict | None = None, timeout: float = 5.0) -> Availability:
    """Probe one endpoint and say what happened, never just False."""
    try:
        _get_json(url, headers, timeout)
        return Availability(True, url)
    except urllib.error.HTTPError as e:
        # A 401/403 means it is *there* and the credential is the problem, which
        # is a different fix from "nothing is listening".
        return Availability(False, f"{url} returned HTTP {e.code}")
    except urllib.error.URLError as e:
        return Availability(False, f"{url} unreachable: {e.reason}")
    except Exception as e:                                  # noqa: BLE001
        return Availability(False, f"{url}: {type(e).__name__}: {e}")


# --------------------------------------------------------------------------
# base
# --------------------------------------------------------------------------

class Backend(ABC):
    """One way of turning a prompt into a shell command.

    `generate` is shared: it times the call, applies the measured parser, and
    reports truncation. Subclasses implement `_complete`, which returns the raw
    string and an optional token count.
    """

    name: str = "base"
    needs_model = True

    def __init__(self, model: str = "", **opts: Any) -> None:
        self.model = model
        self.opts = opts

    @abstractmethod
    def probe(self) -> Availability:
        """Can this run right now? Never raise; say why not."""

    @abstractmethod
    def _complete(self, prompt: str, max_tokens: int, temperature: float
                  ) -> tuple[str, int | None]:
        ...

    def generate(self, prompt: str, *, max_tokens: int = DEFAULT_MAX_TOKENS,
                 temperature: float = 0.0) -> Generation:
        t0 = time.perf_counter()
        raw, n_new = self._complete(prompt, max_tokens, temperature)
        dt = time.perf_counter() - t0
        cmd = parse(raw)
        # A reasoning model that never closed its trace produced no answer, and
        # an empty command with the budget exhausted is that case, not a refusal.
        truncated = bool(n_new and n_new >= max_tokens) and not cmd
        return Generation(command=cmd, raw=raw, backend=self.name, model=self.model,
                          seconds=round(dt, 3), new_tokens=n_new, truncated=truncated)


# --------------------------------------------------------------------------
# the no-model path
# --------------------------------------------------------------------------

class NoBackend(Backend):
    """The default. Ranked documentation, no generation, no dependencies.

    This is not a stub for a missing feature — it is the product's floor, and
    the reason the retrieval tier ships on its own. `generate` is never called
    on it; the CLI checks `needs_model` and prints results instead.
    """

    name = "none"
    needs_model = False

    def probe(self) -> Availability:
        return Availability(True, "no model; ranked documentation only")

    def _complete(self, prompt, max_tokens, temperature):
        raise RuntimeError("NoBackend does not generate; the CLI should not call this")


# --------------------------------------------------------------------------
# local
# --------------------------------------------------------------------------

class OllamaBackend(Backend):
    """Ollama's native API on a local daemon."""

    name = "ollama"
    DEFAULT_HOST = "http://localhost:11434"

    def __init__(self, model: str = "", host: str = "", **opts: Any) -> None:
        super().__init__(model, **opts)
        self.host = (host or os.environ.get("OLLAMA_HOST")
                     or self.DEFAULT_HOST).rstrip("/")
        if not self.host.startswith("http"):
            self.host = f"http://{self.host}"

    def probe(self) -> Availability:
        av = _reach(f"{self.host}/api/tags")
        if not av:
            return Availability(False, f"{av.detail} — is `ollama serve` running?")
        try:
            tags = _get_json(f"{self.host}/api/tags")
            have = [m["name"] for m in tags.get("models", [])]
        except Exception:                                   # noqa: BLE001
            return Availability(True, f"{self.host} (model list unavailable)")
        if not have:
            return Availability(False, f"{self.host} has no models — `ollama pull <model>`")
        if self.model and self.model not in have and f"{self.model}:latest" not in have:
            return Availability(False,
                                f"{self.model!r} not pulled; have {', '.join(have[:6])}")
        return Availability(True, f"{self.host} ({len(have)} models)")

    def _complete(self, prompt, max_tokens, temperature):
        r = _post_json(f"{self.host}/api/generate", {
            "model": self.model, "prompt": prompt, "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        })
        return r.get("response", ""), r.get("eval_count")


class LlamaCppBackend(Backend):
    """A GGUF loaded in-process through `llama-cpp-python`.

    The measured lane. `nl2sh-instantiate/run_gen_gguf.py` scored Nemotron 3
    Nano and Gemma 4 E2B this way, and the chat template comes from the GGUF
    itself rather than a hardcoded fallback.
    """

    name = "llamacpp"

    def __init__(self, model: str = "", n_ctx: int = 4096, n_threads: int = 0,
                 **opts: Any) -> None:
        super().__init__(model, **opts)
        self.n_ctx = n_ctx
        self.n_threads = n_threads or (os.cpu_count() or 4)
        self._llm = None

    def probe(self) -> Availability:
        try:
            import llama_cpp                                # noqa: F401
        except ImportError:
            return Availability(False, "pip install llama-cpp-python "
                                       "(prebuilt CPU wheels: "
                                       "--extra-index-url "
                                       "https://abetlen.github.io/llama-cpp-python/whl/cpu)")
        if not self.model:
            return Availability(False, "no GGUF given; --model is a path or repo:file")
        if ":" not in self.model and not os.path.exists(self.model):
            return Availability(False, f"{self.model!r} is not a file; "
                                       "use a path or 'hf-repo:file.gguf'")
        return Availability(True, f"llama.cpp, {self.n_threads} threads")

    def _load(self):
        if self._llm is not None:
            return self._llm
        from llama_cpp import Llama
        common = dict(n_ctx=self.n_ctx, n_threads=self.n_threads,
                      n_threads_batch=self.n_threads, verbose=False)
        if os.path.exists(self.model):
            self._llm = Llama(model_path=self.model, **common)
        else:                                    # 'org/repo:filename.gguf'
            repo, _, fname = self.model.partition(":")
            self._llm = Llama.from_pretrained(
                repo_id=repo, filename=fname,
                cache_dir=os.path.expanduser("~/.cache/nl2sh/gguf"), **common)
        return self._llm

    def _complete(self, prompt, max_tokens, temperature):
        r = self._load().create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=temperature, repeat_penalty=1.0)
        return (r["choices"][0]["message"]["content"] or "",
                r.get("usage", {}).get("completion_tokens"))


class MLXBackend(Backend):
    """`mlx-lm` on Apple Silicon.

    Untested here — this container is x86 Linux with no Metal. The probe says
    so rather than pretending; a number from this backend needs measuring on a
    Mac before it goes in any table.
    """

    name = "mlx"

    def probe(self) -> Availability:
        try:
            import mlx.core as mx                           # noqa: F401
            import mlx_lm                                   # noqa: F401
        except ImportError:
            return Availability(False, "pip install mlx-lm (Apple Silicon only)")
        if not self.model:
            return Availability(False, "no model given; --model is an mlx-community repo id")
        return Availability(True, "mlx-lm")

    def _complete(self, prompt, max_tokens, temperature):
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler
        model, tok = load(self.model)
        msgs = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        out = generate(model, tok, prompt=text, max_tokens=max_tokens,
                       sampler=make_sampler(temp=temperature), verbose=False)
        return out, None


class TransformersBackend(Backend):
    """Hugging Face `transformers` in-process — the stage-1 and stage-2 harness.

    Slowest of the local options on CPU and the only one that reproduces the
    committed numbers byte for byte, which is why it is here at all.
    """

    name = "transformers"

    def __init__(self, model: str = "", dtype: str = "bfloat16", **opts: Any) -> None:
        super().__init__(model, **opts)
        self.dtype = dtype
        self._pair = None

    def probe(self) -> Availability:
        try:
            import torch, transformers                      # noqa: F401
        except ImportError:
            return Availability(False, "pip install transformers torch")
        if not self.model:
            return Availability(False, "no model given; --model is a HF repo id")
        return Availability(True, f"transformers, {self.dtype}")

    def _load(self):
        if self._pair is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(self.model)
            mdl = AutoModelForCausalLM.from_pretrained(
                self.model, dtype=getattr(torch, self.dtype)).eval()
            self._pair = (tok, mdl)
        return self._pair

    def _complete(self, prompt, max_tokens, temperature):
        import torch
        tok, mdl = self._load()
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            out = mdl.generate(**ids, max_new_tokens=max_tokens,
                               do_sample=temperature > 0,
                               temperature=temperature or None,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
        n_prompt = ids["input_ids"].shape[1]
        return tok.decode(out[0][n_prompt:], skip_special_tokens=True), \
            int(out.shape[1] - n_prompt)


# --------------------------------------------------------------------------
# remote
# --------------------------------------------------------------------------

class OpenAICompatBackend(Backend):
    """Any `/v1/chat/completions` endpoint.

    One class covers OpenAI, `llama-server`, LM Studio, vLLM, Together, Groq
    and OpenRouter, because they all speak the same route. Point `--base-url`
    at the server and set the matching key.
    """

    name = "openai"
    DEFAULT_BASE = "https://api.openai.com/v1"
    ENV_KEY = "OPENAI_API_KEY"

    def __init__(self, model: str = "", base_url: str = "", api_key: str = "",
                 **opts: Any) -> None:
        super().__init__(model, **opts)
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL")
                         or self.DEFAULT_BASE).rstrip("/")
        self.api_key = api_key or os.environ.get(self.ENV_KEY, "")

    def _headers(self) -> dict:
        # A local llama-server ignores the header; sending a placeholder keeps
        # one code path for both.
        return {"Authorization": f"Bearer {self.api_key or 'sk-no-key-required'}"}

    def probe(self) -> Availability:
        local = "localhost" in self.base_url or "127.0.0.1" in self.base_url
        if not self.api_key and not local:
            return Availability(False, f"set {self.ENV_KEY} (or pass --api-key)")
        av = _reach(f"{self.base_url}/models", self._headers())
        if not av and local:
            return Availability(False, f"{av.detail} — is the local server running?")
        return av if av else Availability(False, av.detail)

    def _complete(self, prompt, max_tokens, temperature):
        r = _post_json(f"{self.base_url}/chat/completions", {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature,
        }, self._headers())
        return (r["choices"][0]["message"].get("content") or "",
                r.get("usage", {}).get("completion_tokens"))


class LlamaServerBackend(OpenAICompatBackend):
    """`llama-server` from llama.cpp, which serves the OpenAI route locally."""

    name = "llama-server"
    DEFAULT_BASE = "http://localhost:8080/v1"


class LMStudioBackend(OpenAICompatBackend):
    """LM Studio's local server, same route, different port."""

    name = "lmstudio"
    DEFAULT_BASE = "http://localhost:1234/v1"


class AnthropicBackend(Backend):
    """Claude through the Messages API, over stdlib HTTP."""

    name = "anthropic"
    BASE = "https://api.anthropic.com/v1/messages"
    VERSION = "2023-06-01"

    def __init__(self, model: str = "claude-haiku-4-5-20251001", api_key: str = "",
                 **opts: Any) -> None:
        super().__init__(model, **opts)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    def probe(self) -> Availability:
        if not self.api_key:
            return Availability(False, "set ANTHROPIC_API_KEY (or pass --api-key)")
        return Availability(True, f"anthropic, {self.model}")

    def _complete(self, prompt, max_tokens, temperature):
        r = _post_json(self.BASE, {
            "model": self.model, "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }, {"x-api-key": self.api_key, "anthropic-version": self.VERSION})
        text = "".join(b.get("text", "") for b in r.get("content", [])
                       if b.get("type") == "text")
        return text, r.get("usage", {}).get("output_tokens")


class GeminiBackend(Backend):
    """Gemini through `generateContent`.

    `nl2sh-selfhist/ARCHITECTURE.md` measured `gemini-3.5-flash-lite` at 0.771
    direct, with no retrieval tier at all. That number is the honest ceiling
    this whole local line is trading against, so the backend that produces it
    belongs in the box.
    """

    name = "gemini"
    BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, model: str = "gemini-3.5-flash-lite", api_key: str = "",
                 **opts: Any) -> None:
        super().__init__(model, **opts)
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")

    def probe(self) -> Availability:
        if not self.api_key:
            return Availability(False, "set GEMINI_API_KEY (or pass --api-key)")
        return Availability(True, f"gemini, {self.model}")

    def _complete(self, prompt, max_tokens, temperature):
        r = _post_json(
            f"{self.BASE}/{self.model}:generateContent",
            {"contents": [{"parts": [{"text": prompt}]}],
             "generationConfig": {"temperature": temperature,
                                  "maxOutputTokens": max_tokens}},
            {"x-goog-api-key": self.api_key})
        parts = (r.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts)
        usage = r.get("usageMetadata", {})
        return text, usage.get("candidatesTokenCount")


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

REGISTRY: dict[str, type[Backend]] = {
    "none": NoBackend,
    "ollama": OllamaBackend,
    "llamacpp": LlamaCppBackend,
    "llama-server": LlamaServerBackend,
    "lmstudio": LMStudioBackend,
    "mlx": MLXBackend,
    "transformers": TransformersBackend,
    "openai": OpenAICompatBackend,
    "anthropic": AnthropicBackend,
    "gemini": GeminiBackend,
}

LOCAL = ("ollama", "llamacpp", "llama-server", "lmstudio", "mlx", "transformers")
REMOTE = ("openai", "anthropic", "gemini")


def build(name: str, **opts: Any) -> Backend:
    if name not in REGISTRY:
        raise KeyError(f"unknown backend {name!r}; have {', '.join(REGISTRY)}")
    return REGISTRY[name](**opts)


def survey(models: dict[str, str] | None = None) -> list[tuple[str, Availability]]:
    """Probe every backend once, for `nl2sh doctor`.

    Probes are cheap and must not raise, so one broken backend cannot hide the
    rest — an exception is reported as that backend's unavailability.
    """
    models = models or {}
    out = []
    for name, cls in REGISTRY.items():
        try:
            out.append((name, cls(model=models.get(name, "")).probe()))
        except Exception as e:                              # noqa: BLE001
            out.append((name, Availability(False, f"{type(e).__name__}: {e}")))
    return out

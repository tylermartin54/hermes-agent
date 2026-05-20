"""Theater detector — implementation.

Plugin hook callbacks:
    * ``transform_llm_output_callback`` — fires once per turn, post-tool-loop,
      pre-delivery. Returns the response unchanged (None) on PASS, a prefixed
      string on DELIVER_WITH_FLAG, or a rewritten string on REGENERATE_WITH_FIXES.
    * ``on_session_end_callback`` — clears the per-session regeneration counter
      so a long-lived process doesn't leak memory across sessions.

Seams exposed for testing:
    * ``_run_verifier`` — monkeypatched in unit tests to return a canned verdict.
    * ``_run_rewriter`` — monkeypatched in unit tests to return a canned rewrite.
    * ``_load_detector_config`` — monkeypatched in unit tests to inject config.

The hook itself never raises. Any internal exception falls through to "pass
the response unchanged" + a logger.warning, matching the existing
``transform_llm_output`` contract at ``run_agent.py:15528-15529``.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from .verifier_schema import (
    CHECK_DEFINITIONS,
    FLAG_ONLY_CHECKS,
    REGEN_ELIGIBLE_CHECKS,
    REWRITE_PROMPT,
    VERIFIER_PROMPT,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults + config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "max_regenerations": 2,
    "verifier_timeout_seconds": 8,
    "substantive_min_chars": 200,
    "fire_on_completion_words": True,
    "verifier_provider": None,
    "verifier_model": None,
}

# Words/markers that force the hook to fire regardless of length.
# Lowercase for case-insensitive matching except ANCHOR CHECK which is
# canonically uppercase.
_COMPLETION_KEYWORDS = (
    "done",
    "merged",
    "passed",
    "shipped",
    "verified",
    "fixed",
    "complete",
    "ready",
    "live",
)
_CODE_FENCE = "```"
_ANCHOR_MARKER = "ANCHOR CHECK"


def _load_detector_config() -> dict[str, Any]:
    """Load ``theater_detector`` config from ~/.hermes/config.yaml, merged with defaults."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        user_cfg = cfg.get("theater_detector", {}) if isinstance(cfg, dict) else {}
        if not isinstance(user_cfg, dict):
            return dict(DEFAULT_CONFIG)
        merged = dict(DEFAULT_CONFIG)
        merged.update(user_cfg)
        return merged
    except Exception as exc:
        logger.debug("theater_detector: config load failed (%s); using defaults", exc)
        return dict(DEFAULT_CONFIG)


# ---------------------------------------------------------------------------
# Substantive-response filter
# ---------------------------------------------------------------------------


def _is_substantive(text: str, cfg: dict[str, Any]) -> bool:
    """Per packet §3.3 — OR-chain on char-count + completion keywords + code fences + ANCHOR CHECK."""
    if not text:
        return False
    if len(text) >= cfg["substantive_min_chars"]:
        return True
    lower = text.lower()
    if cfg.get("fire_on_completion_words", True):
        for kw in _COMPLETION_KEYWORDS:
            if kw in lower:
                return True
    if _CODE_FENCE in text:
        return True
    if _ANCHOR_MARKER in text:
        return True
    return False


# ---------------------------------------------------------------------------
# Per-session regeneration counter
# ---------------------------------------------------------------------------

_REGEN_COUNTERS: dict[str, int] = {}


def _get_regen_counter(session_id: str) -> int:
    return _REGEN_COUNTERS.get(session_id or "", 0)


def _increment_regen_counter(session_id: str) -> None:
    key = session_id or ""
    _REGEN_COUNTERS[key] = _REGEN_COUNTERS.get(key, 0) + 1


def _reset_regen_counter(session_id: str) -> None:
    _REGEN_COUNTERS.pop(session_id or "", None)


def reset_all_counters_for_tests() -> None:
    """Test-only helper — clear all per-session state between tests."""
    _REGEN_COUNTERS.clear()


# ---------------------------------------------------------------------------
# Verifier / rewriter LLM calls (BOOT.md credential pattern)
# ---------------------------------------------------------------------------


def _build_openai_client(cfg: dict[str, Any]):
    """Build an OpenAI-compatible client using BOOT.md pattern.

    Resolves credentials via ``gateway.run._resolve_runtime_agent_kwargs`` so
    we reuse the user's active provider lane. Returns None if credentials
    cannot be resolved — caller treats this as verifier failure.
    """
    try:
        from openai import OpenAI
    except Exception as exc:
        logger.debug("theater_detector: openai SDK not importable: %s", exc)
        return None

    api_key = None
    base_url = None
    try:
        from gateway.run import _resolve_runtime_agent_kwargs

        runtime_kwargs = _resolve_runtime_agent_kwargs() or {}
        api_key = runtime_kwargs.get("api_key")
        base_url = runtime_kwargs.get("base_url")
    except Exception as exc:
        logger.debug("theater_detector: BOOT.md resolve failed: %s", exc)

    try:
        client_kwargs: dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        return OpenAI(**client_kwargs)
    except Exception as exc:
        logger.debug("theater_detector: OpenAI client construction failed: %s", exc)
        return None


def _resolve_verifier_model(cfg: dict[str, Any]) -> str:
    """Verifier model — explicit override OR gateway default OR safe fallback."""
    explicit = cfg.get("verifier_model")
    if explicit:
        return str(explicit)
    try:
        from gateway.run import _resolve_gateway_model

        m = _resolve_gateway_model()
        if m:
            return m
    except Exception:
        pass
    return "claude-sonnet-4-6"


def _run_verifier(response_text: str, cfg: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Call the verifier LLM. Return parsed verdict dict, or None on failure.

    This is the seam that unit tests monkeypatch. The implementation is
    deliberately conservative: any exception → return None → hook passes
    response through unchanged. No theater is shipped if the verifier is
    broken; we don't fail closed by injecting [VERIFIER FLAGGED] on every
    response.
    """
    client = _build_openai_client(cfg)
    if client is None:
        return None
    model = _resolve_verifier_model(cfg)
    timeout = cfg.get("verifier_timeout_seconds", 8)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": VERIFIER_PROMPT},
                {"role": "user", "content": f"DRAFT RESPONSE TO VERIFY:\n\n{response_text}"},
            ],
            timeout=timeout,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.warning("theater_detector: verifier call failed: %s", exc)
        return None

    return _parse_verdict_json(raw)


def _parse_verdict_json(raw: str) -> Optional[dict[str, Any]]:
    """Parse the verifier's JSON output. Tolerates leading/trailing prose."""
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        # Tolerate the verifier wrapping the JSON in prose or fences.
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(raw[start : end + 1])
        except Exception as exc:
            logger.warning("theater_detector: could not parse verdict JSON: %s", exc)
            return None


def _run_rewriter(
    response_text: str, verdict: dict[str, Any], cfg: dict[str, Any]
) -> Optional[str]:
    """Call the rewriter LLM to clean text-style violations. Return None on failure."""
    client = _build_openai_client(cfg)
    if client is None:
        return None
    model = _resolve_verifier_model(cfg)
    timeout = cfg.get("verifier_timeout_seconds", 8)

    violations_str = "\n".join(
        f"- {v.get('check')}: {v.get('evidence', '')}"
        for v in verdict.get("violations", [])
    )
    prompt = REWRITE_PROMPT.format(violations=violations_str, draft=response_text)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            timeout=timeout,
        )
        out = (resp.choices[0].message.content or "").strip()
        return out or None
    except Exception as exc:
        logger.warning("theater_detector: rewriter call failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------


def _normalize_verdict(verdict: dict[str, Any]) -> str:
    """Promote DELIVER_WITH_FLAG when any FLAG_ONLY check is present.

    Defensive: the verifier MAY return REGENERATE_WITH_FIXES with structural
    violations (a model can mis-classify). We re-classify based on the actual
    violations list so the verdict is mechanically consistent.
    """
    requested = verdict.get("verdict", "PASS")
    violations = verdict.get("violations") or []
    if not violations:
        return "PASS"
    checks_seen = {str(v.get("check")) for v in violations if isinstance(v, dict)}
    if checks_seen & FLAG_ONLY_CHECKS:
        return "DELIVER_WITH_FLAG"
    if checks_seen & REGEN_ELIGIBLE_CHECKS and requested == "REGENERATE_WITH_FIXES":
        return "REGENERATE_WITH_FIXES"
    if requested in {"PASS", "REGENERATE_WITH_FIXES", "DELIVER_WITH_FLAG"}:
        return requested
    return "PASS"


def _format_flag_prefix(verdict: dict[str, Any], suffix: str = "") -> str:
    violations = verdict.get("violations") or []
    names = ",".join(
        sorted({str(v.get("check")) for v in violations if isinstance(v, dict) and v.get("check")})
    )
    if not names:
        names = "unspecified"
    tag = f"[VERIFIER FLAGGED: {names}{suffix}]"
    return tag


# ---------------------------------------------------------------------------
# Hook callbacks
# ---------------------------------------------------------------------------


def transform_llm_output_callback(**kwargs: Any) -> Optional[str]:
    """Plugin hook callback. Returns str to replace text, None to pass through.

    Per the ``transform_llm_output`` contract: first non-empty string wins
    across all plugins, so returning None on PASS preserves other plugins'
    transforms (e.g. spongebob mode, redaction). We only return a string when
    we want to actively replace the response.
    """
    response_text = kwargs.get("response_text", "")
    session_id = kwargs.get("session_id", "") or ""

    if not response_text:
        return None

    try:
        cfg = _load_detector_config()
    except Exception as exc:  # defensive — config load shouldn't throw
        logger.debug("theater_detector: config load raised: %s", exc)
        return None

    if not cfg.get("enabled", True):
        return None

    if not _is_substantive(response_text, cfg):
        return None

    counter = _get_regen_counter(session_id)
    max_regen = int(cfg.get("max_regenerations", 2))

    try:
        verdict = _run_verifier(response_text, cfg)
    except Exception as exc:
        logger.warning("theater_detector: verifier raised: %s", exc)
        return None

    if not verdict or not isinstance(verdict, dict):
        # Verifier failed → pass through. We do NOT fail closed.
        return None

    normalized = _normalize_verdict(verdict)

    if normalized == "PASS":
        return None

    if normalized == "DELIVER_WITH_FLAG":
        return f"{_format_flag_prefix(verdict)}\n\n{response_text}"

    if normalized == "REGENERATE_WITH_FIXES":
        if counter >= max_regen:
            return f"{_format_flag_prefix(verdict, ' (max regens exceeded)')}\n\n{response_text}"
        _increment_regen_counter(session_id)
        try:
            rewritten = _run_rewriter(response_text, verdict, cfg)
        except Exception as exc:
            logger.warning("theater_detector: rewriter raised: %s", exc)
            rewritten = None
        if rewritten and rewritten.strip():
            return rewritten
        return f"{_format_flag_prefix(verdict, ' (regen failed)')}\n\n{response_text}"

    # Unknown verdict — pass through (defensive).
    return None


def on_session_end_callback(**kwargs: Any) -> None:
    """Clear the per-session regen counter when a session ends."""
    session_id = kwargs.get("session_id", "") or ""
    _reset_regen_counter(session_id)

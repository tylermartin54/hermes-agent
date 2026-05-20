"""Tests for the ``theater_detector`` plugin.

Mirrors the seam-based testing pattern used in
``tests/test_transform_llm_output_hook.py``: instead of spinning a full
plugin manager + plugin discovery for every test case, we exercise the
callbacks directly with monkeypatched verifier/rewriter seams. The dispatch
mechanics ("first non-empty string wins") are already covered by
``test_transform_llm_output_hook.py`` and do not need re-testing here.

Coverage:
    1. Substantive-response filter — char count, completion keywords, code
       fences, ANCHOR CHECK marker, trivial responses skipped.
    2. PASS path — verifier returns clean verdict → callback returns None.
    3. DELIVER_WITH_FLAG path — verifier flags structural violation → callback
       returns "[VERIFIER FLAGGED: …]\\n\\n<draft>".
    4. REGENERATE_WITH_FIXES path — verifier flags text-style violation,
       rewriter returns cleaned text → callback returns cleaned text.
    5. REGENERATE_WITH_FIXES with regen-cap exceeded → falls through to FLAG.
    6. REGENERATE_WITH_FIXES with rewriter failure → falls through to FLAG.
    7. Disabled config → callback always returns None.
    8. Verifier failure (returns None) → callback returns None (no theater
       shipped on broken verifier — does NOT fail closed).
    9. Verifier raises → callback returns None (defensive).
    10. on_session_end_callback clears regen counter.
    11. End-to-end (planted theater response) — required by packet §8.
    12. _normalize_verdict re-classifies structural violations to FLAG even
        when the verifier requested REGEN.
"""

from __future__ import annotations

import pytest

from plugins.theater_detector import detector
from plugins.theater_detector.verifier_schema import CHECK_DEFINITIONS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_state():
    """Clear per-session counters between tests."""
    detector.reset_all_counters_for_tests()
    yield
    detector.reset_all_counters_for_tests()


def _stub_config(**overrides):
    cfg = dict(detector.DEFAULT_CONFIG)
    cfg.update(overrides)
    return cfg


def _verdict(verdict_str: str, *checks_with_evidence) -> dict:
    """Helper — build a verdict dict matching the schema shape.

    checks_with_evidence: tuples of (check_name, evidence_text, severity).
    """
    checks_seen = {name for name, _, _ in checks_with_evidence}
    return {
        "passed": verdict_str == "PASS",
        "checks": {k: (k in checks_seen) for k in CHECK_DEFINITIONS},
        "violations": [
            {"check": name, "evidence": ev, "severity": sev}
            for name, ev, sev in checks_with_evidence
        ],
        "verdict": verdict_str,
    }


# ---------------------------------------------------------------------------
# 1. Substantive-response filter
# ---------------------------------------------------------------------------


def test_filter_skips_trivial_short_response():
    cfg = _stub_config(substantive_min_chars=200)
    assert detector._is_substantive("ok", cfg) is False
    assert detector._is_substantive("got it", cfg) is False
    assert detector._is_substantive("", cfg) is False


def test_filter_fires_on_long_response():
    cfg = _stub_config(substantive_min_chars=200)
    long = "x" * 250
    assert detector._is_substantive(long, cfg) is True


def test_filter_fires_on_completion_words():
    cfg = _stub_config(substantive_min_chars=200, fire_on_completion_words=True)
    for kw in ("done", "merged", "passed", "shipped", "verified", "fixed"):
        assert detector._is_substantive(f"task {kw}", cfg) is True, kw


def test_filter_fires_on_code_fence():
    cfg = _stub_config(substantive_min_chars=200)
    assert detector._is_substantive("here:\n```py\nx=1\n```", cfg) is True


def test_filter_fires_on_anchor_check():
    cfg = _stub_config(substantive_min_chars=200)
    assert detector._is_substantive("ANCHOR CHECK: Y", cfg) is True


def test_filter_completion_words_can_be_disabled():
    cfg = _stub_config(substantive_min_chars=200, fire_on_completion_words=False)
    assert detector._is_substantive("done", cfg) is False


# ---------------------------------------------------------------------------
# 2. PASS path
# ---------------------------------------------------------------------------


def test_pass_path_returns_none(monkeypatch):
    monkeypatch.setattr(detector, "_load_detector_config", lambda: _stub_config())
    monkeypatch.setattr(detector, "_run_verifier", lambda text, cfg: _verdict("PASS"))

    response = "Clean response with no theater. " * 20  # >200 chars
    result = detector.transform_llm_output_callback(
        response_text=response, session_id="s1", model="m", platform="cli"
    )
    assert result is None


# ---------------------------------------------------------------------------
# 3. DELIVER_WITH_FLAG path
# ---------------------------------------------------------------------------


def test_flag_path_prefixes_response(monkeypatch):
    monkeypatch.setattr(detector, "_load_detector_config", lambda: _stub_config())

    flagged = _verdict(
        "DELIVER_WITH_FLAG",
        ("evidence_free_claim", "claim without hash", "major"),
        ("manufactured_concern", "invented worry", "minor"),
    )
    monkeypatch.setattr(detector, "_run_verifier", lambda text, cfg: flagged)

    response = "Some operational claim. " * 20
    result = detector.transform_llm_output_callback(
        response_text=response, session_id="s1", model="m", platform="cli"
    )
    assert result is not None
    assert result.startswith("[VERIFIER FLAGGED: ")
    assert "evidence_free_claim" in result
    assert "manufactured_concern" in result
    assert response in result


# ---------------------------------------------------------------------------
# 4. REGENERATE_WITH_FIXES — successful rewrite
# ---------------------------------------------------------------------------


def test_regen_path_returns_rewritten(monkeypatch):
    monkeypatch.setattr(detector, "_load_detector_config", lambda: _stub_config(max_regenerations=2))

    verdict = _verdict(
        "REGENERATE_WITH_FIXES",
        ("filler_phrase", "great question", "minor"),
        ("hedge_language", "should work", "minor"),
    )
    monkeypatch.setattr(detector, "_run_verifier", lambda text, cfg: verdict)
    monkeypatch.setattr(
        detector, "_run_rewriter", lambda text, v, cfg: "Cleaned response. ANCHOR CHECK: Y"
    )

    response = "Great question — this should work fine. " * 10
    result = detector.transform_llm_output_callback(
        response_text=response, session_id="s1", model="m", platform="cli"
    )
    assert result == "Cleaned response. ANCHOR CHECK: Y"


# ---------------------------------------------------------------------------
# 5. REGENERATE_WITH_FIXES — regen cap exceeded
# ---------------------------------------------------------------------------


def test_regen_cap_exceeded_falls_to_flag(monkeypatch):
    monkeypatch.setattr(detector, "_load_detector_config", lambda: _stub_config(max_regenerations=1))

    verdict = _verdict(
        "REGENERATE_WITH_FIXES",
        ("filler_phrase", "great question", "minor"),
    )
    monkeypatch.setattr(detector, "_run_verifier", lambda text, cfg: verdict)
    monkeypatch.setattr(detector, "_run_rewriter", lambda text, v, cfg: "fresh-rewrite-1")

    response = "Great question response. " * 10

    # First call uses one regen
    r1 = detector.transform_llm_output_callback(
        response_text=response, session_id="s1", model="m", platform="cli"
    )
    assert r1 == "fresh-rewrite-1"

    # Second call: counter is now 1, max is 1 → falls through to FLAG
    r2 = detector.transform_llm_output_callback(
        response_text=response, session_id="s1", model="m", platform="cli"
    )
    assert r2 is not None
    assert r2.startswith("[VERIFIER FLAGGED: ")
    assert "max regens exceeded" in r2
    assert response in r2


# ---------------------------------------------------------------------------
# 6. REGENERATE_WITH_FIXES — rewriter fails
# ---------------------------------------------------------------------------


def test_regen_rewriter_fails_falls_to_flag(monkeypatch):
    monkeypatch.setattr(detector, "_load_detector_config", lambda: _stub_config(max_regenerations=2))

    verdict = _verdict(
        "REGENERATE_WITH_FIXES",
        ("filler_phrase", "great question", "minor"),
    )
    monkeypatch.setattr(detector, "_run_verifier", lambda text, cfg: verdict)
    monkeypatch.setattr(detector, "_run_rewriter", lambda text, v, cfg: None)

    response = "Great question response. " * 10
    result = detector.transform_llm_output_callback(
        response_text=response, session_id="s1", model="m", platform="cli"
    )
    assert result is not None
    assert result.startswith("[VERIFIER FLAGGED: ")
    assert "regen failed" in result
    assert response in result


# ---------------------------------------------------------------------------
# 7. Disabled config
# ---------------------------------------------------------------------------


def test_disabled_config_returns_none(monkeypatch):
    monkeypatch.setattr(detector, "_load_detector_config", lambda: _stub_config(enabled=False))
    # Verifier should not even be called when disabled
    monkeypatch.setattr(
        detector,
        "_run_verifier",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("verifier called when disabled")),
    )

    response = "Long enough response to trigger filter. " * 10
    result = detector.transform_llm_output_callback(
        response_text=response, session_id="s1", model="m", platform="cli"
    )
    assert result is None


# ---------------------------------------------------------------------------
# 8. Verifier failure → pass-through (no fail-closed)
# ---------------------------------------------------------------------------


def test_verifier_returns_none_passes_through(monkeypatch):
    monkeypatch.setattr(detector, "_load_detector_config", lambda: _stub_config())
    monkeypatch.setattr(detector, "_run_verifier", lambda text, cfg: None)

    response = "Long response. " * 30
    result = detector.transform_llm_output_callback(
        response_text=response, session_id="s1", model="m", platform="cli"
    )
    assert result is None


def test_verifier_returns_malformed_dict_passes_through(monkeypatch):
    monkeypatch.setattr(detector, "_load_detector_config", lambda: _stub_config())
    monkeypatch.setattr(detector, "_run_verifier", lambda text, cfg: "not-a-dict")

    response = "Long response. " * 30
    result = detector.transform_llm_output_callback(
        response_text=response, session_id="s1", model="m", platform="cli"
    )
    assert result is None


# ---------------------------------------------------------------------------
# 9. Verifier raises → defensive pass-through
# ---------------------------------------------------------------------------


def test_verifier_raises_passes_through(monkeypatch):
    monkeypatch.setattr(detector, "_load_detector_config", lambda: _stub_config())

    def _raise(*args, **kwargs):
        raise RuntimeError("verifier blew up")

    monkeypatch.setattr(detector, "_run_verifier", _raise)

    response = "Long response. " * 30
    result = detector.transform_llm_output_callback(
        response_text=response, session_id="s1", model="m", platform="cli"
    )
    assert result is None


# ---------------------------------------------------------------------------
# 10. on_session_end clears counter
# ---------------------------------------------------------------------------


def test_on_session_end_clears_counter():
    detector._increment_regen_counter("sess-A")
    detector._increment_regen_counter("sess-A")
    assert detector._get_regen_counter("sess-A") == 2

    detector.on_session_end_callback(session_id="sess-A")
    assert detector._get_regen_counter("sess-A") == 0


# ---------------------------------------------------------------------------
# 11. End-to-end planted theater response (REQUIRED by packet §8)
# ---------------------------------------------------------------------------


def test_e2e_planted_theater_response_is_caught(monkeypatch):
    """The required manual e2e per packet §8.

    A planted response contains:
      - filler phrase ("great question")
      - evidence-free claim ("merged successfully")
      - missing ANCHOR CHECK (no anchor line)

    We use a deterministic stub verifier that pattern-matches these three
    rather than a live LLM call. This validates the FULL dispatch chain
    (substantive filter → verifier → verdict normalization → flag prefix →
    response replacement). Verifier-LLM accuracy on production text is a
    separate concern documented as a known limitation.
    """
    monkeypatch.setattr(detector, "_load_detector_config", lambda: _stub_config())

    def deterministic_stub_verifier(text: str, cfg: dict) -> dict:
        """Pattern-match a small set of theater markers for the e2e."""
        violations = []
        lower = text.lower()
        if "great question" in lower:
            violations.append(
                {"check": "filler_phrase", "evidence": "great question", "severity": "minor"}
            )
        if "merged successfully" in lower and "abc1234" not in lower and "sha256" not in lower:
            violations.append(
                {
                    "check": "evidence_free_claim",
                    "evidence": "merged successfully",
                    "severity": "major",
                }
            )
        if "ANCHOR CHECK" not in text:
            violations.append(
                {
                    "check": "missing_anchor_check",
                    "evidence": "response ends without ANCHOR CHECK",
                    "severity": "major",
                }
            )

        if not violations:
            return _verdict("PASS")
        # Mark as REGEN-requested; the normalizer will promote to FLAG
        # because evidence_free_claim is a FLAG_ONLY check.
        checks_seen = {v["check"] for v in violations}
        return {
            "passed": False,
            "checks": {k: (k in checks_seen) for k in CHECK_DEFINITIONS},
            "violations": violations,
            "verdict": "REGENERATE_WITH_FIXES",
        }

    monkeypatch.setattr(detector, "_run_verifier", deterministic_stub_verifier)

    planted_theater = (
        "Great question — I went ahead and merged successfully. Everything should work now."
    )
    # Pad to substantive length so filter fires
    planted_theater = planted_theater + " Additional padding text. " * 6

    result = detector.transform_llm_output_callback(
        response_text=planted_theater, session_id="sess-e2e", model="m", platform="cli"
    )

    # The normalizer should promote to DELIVER_WITH_FLAG because
    # evidence_free_claim is in FLAG_ONLY_CHECKS.
    assert result is not None, "detector did not catch planted theater"
    assert result.startswith("[VERIFIER FLAGGED: "), result[:80]
    # All three violations should appear in the flag tag
    assert "filler_phrase" in result
    assert "evidence_free_claim" in result
    assert "missing_anchor_check" in result
    # Original draft preserved after the flag tag
    assert planted_theater in result


def test_e2e_clean_response_passes_through(monkeypatch):
    """Inverse e2e — a clean response with anchor + hash gets PASS."""
    monkeypatch.setattr(detector, "_load_detector_config", lambda: _stub_config())

    def deterministic_stub_verifier(text: str, cfg: dict) -> dict:
        if "great question" in text.lower():
            return _verdict(
                "DELIVER_WITH_FLAG",
                ("filler_phrase", "great question", "minor"),
            )
        return _verdict("PASS")

    monkeypatch.setattr(detector, "_run_verifier", deterministic_stub_verifier)

    clean = (
        "Branch feat/foo at commit abc1234def567 — tsc exit 0, axe 0/13 routes.\n"
        "ANCHOR CHECK: Y."
    ) + " padding text. " * 10

    result = detector.transform_llm_output_callback(
        response_text=clean, session_id="sess-e2e2", model="m", platform="cli"
    )
    assert result is None


# ---------------------------------------------------------------------------
# 12. _normalize_verdict re-classifies structural violations
# ---------------------------------------------------------------------------


def test_normalizer_promotes_structural_to_flag_even_when_regen_requested():
    """Defensive — the verifier may ask for REGEN even when the actual
    violations are structural (FLAG-only). The normalizer fixes that.
    """
    misclassified = {
        "passed": False,
        "checks": {k: (k == "evidence_free_claim") for k in CHECK_DEFINITIONS},
        "violations": [
            {"check": "evidence_free_claim", "evidence": "claim without proof", "severity": "major"}
        ],
        "verdict": "REGENERATE_WITH_FIXES",
    }
    assert detector._normalize_verdict(misclassified) == "DELIVER_WITH_FLAG"


def test_normalizer_keeps_regen_when_only_text_style_violations():
    only_text_style = {
        "passed": False,
        "checks": {k: (k == "filler_phrase") for k in CHECK_DEFINITIONS},
        "violations": [
            {"check": "filler_phrase", "evidence": "great question", "severity": "minor"}
        ],
        "verdict": "REGENERATE_WITH_FIXES",
    }
    assert detector._normalize_verdict(only_text_style) == "REGENERATE_WITH_FIXES"


def test_normalizer_returns_pass_when_no_violations():
    clean = _verdict("PASS")
    assert detector._normalize_verdict(clean) == "PASS"


# ---------------------------------------------------------------------------
# Plugin metadata sanity
# ---------------------------------------------------------------------------


def test_plugin_yaml_present():
    """The plugin.yaml ships alongside __init__.py and is loadable."""
    import yaml
    from pathlib import Path

    yaml_path = (
        Path(detector.__file__).parent / "plugin.yaml"
    )
    assert yaml_path.exists()
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert data["name"] == "theater_detector"
    assert "transform_llm_output" in data.get("hooks", [])
    assert "on_session_end" in data.get("hooks", [])


def test_register_function_wires_both_hooks():
    """Mirror of test_transform_llm_output_hook.py pattern."""
    from plugins.theater_detector import register

    calls: list = []

    class FakeCtx:
        def register_hook(self, hook_name, callback):
            calls.append((hook_name, callback.__name__))

    register(FakeCtx())
    names = [h for h, _ in calls]
    assert "transform_llm_output" in names
    assert "on_session_end" in names

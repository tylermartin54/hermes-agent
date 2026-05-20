"""theater_detector — pre-delivery theater detector plugin for Hermes.

Implements Mechanism 6 (M2.2) of the Hermes-PC Professional Build Protocol:
adversarial verification of Hermes' own response before it is delivered to
the user. Fires on every substantive ``transform_llm_output`` event.

Hooks registered
----------------
- ``transform_llm_output``: scans the draft response for theater patterns and
  may PASS, REGENERATE_WITH_FIXES, or DELIVER_WITH_FLAG.
- ``on_session_end``: clears the per-session regeneration counter.

Config in ~/.hermes/config.yaml (all optional — defaults below):

    theater_detector:
      enabled: true                # default true. False = no-op.
      max_regenerations: 2         # cycles before falling through to DELIVER_WITH_FLAG.
      verifier_timeout_seconds: 8  # LLM verifier call timeout. Per packet §7.
      substantive_min_chars: 200   # responses shorter than this skip the hook unless
                                   # they contain completion-claim keywords / code fences /
                                   # ANCHOR CHECK.
      fire_on_completion_words: true   # if true, completion-claim words always trigger.
      verifier_provider: null      # null = use gateway provider (BOOT.md pattern).
      verifier_model: null         # null = use gateway model.

To opt the plugin in, list it under ``plugins.enabled`` in the same config::

    plugins:
      enabled:
        - theater_detector

Source canon: ``coding-db/tasks/2026-05-18-hermes-agent-theater-detector-t2.8.md``.
"""

from __future__ import annotations

import logging

from .detector import (
    on_session_end_callback,
    transform_llm_output_callback,
)

logger = logging.getLogger(__name__)


def register(ctx):
    """Plugin entry point — wire callbacks into the hook registry."""
    ctx.register_hook("transform_llm_output", transform_llm_output_callback)
    ctx.register_hook("on_session_end", on_session_end_callback)
    logger.info(
        "theater_detector registered (transform_llm_output, on_session_end)"
    )

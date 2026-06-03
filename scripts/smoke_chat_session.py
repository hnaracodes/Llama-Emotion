#!/usr/bin/env python3
"""Non-interactive smoke test: affect refresh + phase_chat.json artifact."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.chat.session import ChatSession
from src.chat.signatures import ema_update, extract_signature_from_pipeline
from src.chat.tone_markers import detect_shift, dominant_tone, format_shift_banner
from src.affective.tribev2_client import pipeline_to_spikes, run_tribev2_from_transcript
from src.config import DELTA_THETA


def main() -> int:
    session = ChatSession()
    session.append("user", "I failed my exam today and I feel awful.")
    session.append("assistant", "That sounds really hard. I'm here with you.")
    session.append("user", "Actually I'm starting to feel a bit hopeful.")

    prev_traits = dict(session.traits)
    fmri, source = run_tribev2_from_transcript(session.transcript_messages())
    pipe = pipeline_to_spikes(fmri, theta=DELTA_THETA)
    sig = extract_signature_from_pipeline(fmri, pipe, device="cpu")
    session.affect_vector = ema_update(None, sig["vector"])
    session.traits = sig["traits"]
    session.dominant_tone = dominant_tone(session.traits)
    session.last_refresh_ts = time.time()

    shifted, mag, before, after = detect_shift(prev_traits, session.traits)
    if shifted:
        session.record_tone_event(
            event="smoke_refresh",
            before=before,
            after=after,
            shift=mag,
            traits=session.traits,
        )
        print(format_shift_banner(before, after, mag))

    out = PROJECT_ROOT / "data" / "artifacts" / "benchmarks" / "phase_chat.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"source": source, **session.to_log_dict()}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    print(f"tone={session.dominant_tone} traits={session.traits}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

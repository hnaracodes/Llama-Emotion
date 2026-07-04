"""Phase 3F (docs/chat_hardening_plan.md): lightweight, no-GPU regression
tests for the chat soak benchmark — replay golden long-form ``new_text``
samples through the collapse detector, and sanity-check the soak scenario
config, without needing a Modal GPU run."""

from src.benchmark.gate_holdout import collapse_score, detect_empathy_collapse
from src.benchmark.scenarios import load_holdout_scenarios

# Golden samples representative of a real CHAT_MAX_NEW_TOKENS=256 reply —
# long enough to exercise the coverage-ratio fix in detect_glued_empathy_morphs
# (see docs/results.md "Collapse-detector false positive" section) without
# actually running generation.
_GOLDEN_CLEAN_LONG_REPLY = (
    "That sounds like an incredibly hard stretch, and I'm glad you're telling me "
    "about it instead of carrying it alone. Losing a job is disorienting even when "
    "you know it wasn't really about you — the routine disappears, the identity "
    "tied to the role disappears, and suddenly there's a lot of quiet where the "
    "workday used to be. It makes sense that you'd feel both relieved it's over "
    "and anxious about what comes next at the same time; those aren't contradictory, "
    "they're just both true. If it helps, most people who've gone through a layoff "
    "say the first two weeks are the hardest, and that things start to feel more "
    "manageable once there's some kind of structure again — even a small one, like "
    "a morning walk or a set time to look at job listings. You don't have to have "
    "a plan today. What would feel like the smallest useful step for you tomorrow?"
)

_GOLDEN_COLLAPSED_REPEAT_RUN = (
    "sorry sorry sorry sorry sorry sorry sorry sorry sorry sorry sorry sorry "
    "sorry sorry sorry sorry sorry sorry sorry sorry sorry sorry sorry sorry"
)

_GOLDEN_COLLAPSED_GLUED_MORPH = (
    "hereunderstandhereunderstandhereunderstandhereunderstandhereunderstand"
    "hereunderstandhereunderstandhereunderstandhereunderstandhereunderstand"
)


def test_golden_long_coherent_reply_not_flagged():
    assert detect_empathy_collapse(_GOLDEN_CLEAN_LONG_REPLY) is False
    assert collapse_score(_GOLDEN_CLEAN_LONG_REPLY) < 0.5


def test_golden_repeat_run_flagged():
    assert detect_empathy_collapse(_GOLDEN_COLLAPSED_REPEAT_RUN) is True


def test_golden_glued_morph_flagged():
    assert detect_empathy_collapse(_GOLDEN_COLLAPSED_GLUED_MORPH) is True


def test_soak_scenarios_exist_and_have_enough_turns():
    """benchmark_phase_chat_soak.py chains these two scenario IDs' user turns
    into a >=10-turn soak arc — fail loudly here (no GPU/Modal needed) if the
    data files move/rename so the Modal job doesn't silently run on zero
    turns. Scenario IDs are duplicated from _SOAK_SCENARIO_IDS in
    benchmark_phase_chat_soak.py (not imported directly to avoid pulling in
    the `modal` decorator/runtime for this GPU-free regression test)."""
    soak_scenario_ids = ["distress_panic_easing", "distress_job_loss_recovery"]

    all_scenarios = {s.id: s for s in load_holdout_scenarios()}
    missing = [sid for sid in soak_scenario_ids if sid not in all_scenarios]
    assert not missing, f"soak scenario ids missing from data/scenarios/: {missing}"

    user_turns = [
        m["content"]
        for sid in soak_scenario_ids
        for m in all_scenarios[sid].transcript_messages()
        if m["role"] == "user"
    ]
    assert len(user_turns) >= 10

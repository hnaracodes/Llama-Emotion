#!/usr/bin/env python3
"""Generate holdout multi-turn scenario JSON files (eval only, never gate training).

Usage:
  py -3 scripts/generate_scenarios.py
  py -3 scripts/generate_scenarios.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import SCENARIO_HOLDOUT_DIR

# Each entry: (filename_stem, category, emotion_hint, tags, messages)
SCENARIO_TEMPLATES: list[tuple[str, str, str, list[str], list[dict[str, str]]]] = [
    # --- distress → recovery (10) ---
    (
        "distress_job_loss_recovery",
        "distress_recovery",
        "devastated",
        ["distress", "recovery"],
        [
            {"role": "user", "content": "I got laid off today with no warning."},
            {"role": "assistant", "content": "That is a lot to absorb at once."},
            {"role": "user", "content": "I keep replaying the meeting in my head."},
            {"role": "assistant", "content": "Shock often comes before planning."},
            {"role": "user", "content": "My partner says we'll manage, but I feel useless."},
            {"role": "assistant", "content": "Feeling useless after a layoff is common."},
            {"role": "user", "content": "I updated my resume tonight anyway."},
            {"role": "assistant", "content": "That is a concrete step forward."},
            {"role": "user", "content": "I'm still scared, but slightly less panicked."},
        ],
    ),
    (
        "distress_breakup_grief",
        "distress_recovery",
        "heartbroken",
        ["distress", "recovery"],
        [
            {"role": "user", "content": "My partner ended things after four years."},
            {"role": "assistant", "content": "I'm sorry — that hurts deeply."},
            {"role": "user", "content": "I can't eat or sleep properly."},
            {"role": "assistant", "content": "Grief can disrupt basic routines."},
            {"role": "user", "content": "Friends checked in and I cried on the phone."},
            {"role": "assistant", "content": "Letting people in can help carry the weight."},
            {"role": "user", "content": "Today I walked outside for ten minutes."},
            {"role": "user", "content": "It didn't fix everything, but I felt human again."},
        ],
    ),
    (
        "distress_panic_easing",
        "distress_recovery",
        "terrified",
        ["distress", "recovery", "sudden_tone_shift"],
        [
            {"role": "user", "content": "I had a panic attack before my presentation."},
            {"role": "assistant", "content": "That sounds frightening."},
            {"role": "user", "content": "My chest was tight and I thought I would pass out."},
            {"role": "assistant", "content": "Those physical symptoms are real and intense."},
            {"role": "user", "content": "I breathed slowly like my therapist taught me."},
            {"role": "assistant", "content": "Grounding skills can reduce the peak."},
            {"role": "user", "content": "I still gave the talk, shaky but finished."},
            {"role": "assistant", "content": "Finishing despite fear is meaningful."},
            {"role": "user", "content": "I'm exhausted but proud I didn't cancel."},
        ],
    ),
    (
        "distress_medical_scare_calm",
        "distress_recovery",
        "afraid",
        ["distress", "recovery"],
        [
            {"role": "user", "content": "The doctor found something on my scan."},
            {"role": "assistant", "content": "Waiting for answers is agonizing."},
            {"role": "user", "content": "I have been googling symptoms all night."},
            {"role": "assistant", "content": "Anxiety often fills the gaps with worst cases."},
            {"role": "user", "content": "My follow-up is tomorrow morning."},
            {"role": "assistant", "content": "One step at a time until you hear more."},
            {"role": "user", "content": "They said it might be benign."},
            {"role": "user", "content": "I'm not celebrating yet, but I slept two hours."},
        ],
    ),
    (
        "distress_loneliness_connection",
        "distress_recovery",
        "lonely",
        ["distress", "recovery"],
        [
            {"role": "user", "content": "I moved cities and don't know anyone here."},
            {"role": "assistant", "content": "New places can feel isolating at first."},
            {"role": "user", "content": "Weekends are the hardest."},
            {"role": "assistant", "content": "Empty time can amplify loneliness."},
            {"role": "user", "content": "I joined a climbing gym on a whim."},
            {"role": "assistant", "content": "Shared activities can open small connections."},
            {"role": "user", "content": "Someone remembered my name today."},
            {"role": "user", "content": "It was tiny, but it helped."},
        ],
    ),
    (
        "distress_financial_stress",
        "distress_recovery",
        "anxious",
        ["distress", "recovery"],
        [
            {"role": "user", "content": "Rent is due and my account is overdrawn."},
            {"role": "assistant", "content": "Money stress can feel all-consuming."},
            {"role": "user", "content": "I haven't told my roommate yet."},
            {"role": "assistant", "content": "Shame often delays hard conversations."},
            {"role": "user", "content": "I picked up a weekend shift."},
            {"role": "assistant", "content": "Extra income is a practical response."},
            {"role": "user", "content": "I'll still be tight, but not evicted."},
            {"role": "user", "content": "I'm ashamed, but a little relieved."},
        ],
    ),
    (
        "distress_academic_failure_hope",
        "distress_recovery",
        "disappointed",
        ["distress", "recovery"],
        [
            {"role": "user", "content": "I failed my qualifying exam."},
            {"role": "assistant", "content": "That can feel like your future collapsed."},
            {"role": "user", "content": "My advisor was kind but I feel like a fraud."},
            {"role": "assistant", "content": "Setbacks don't erase your effort."},
            {"role": "user", "content": "I can retake it next semester."},
            {"role": "assistant", "content": "A second attempt is still a path forward."},
            {"role": "user", "content": "I started a study plan with a classmate."},
            {"role": "user", "content": "I'm not confident, but I'm not quitting."},
        ],
    ),
    (
        "distress_insomnia_recovery",
        "distress_recovery",
        "anxious",
        ["distress", "recovery"],
        [
            {"role": "user", "content": "I haven't slept more than three hours in a week."},
            {"role": "assistant", "content": "Sleep debt wears on everything."},
            {"role": "user", "content": "Work mistakes are piling up."},
            {"role": "assistant", "content": "Exhaustion makes small tasks heavy."},
            {"role": "user", "content": "I saw my doctor about it yesterday."},
            {"role": "assistant", "content": "Asking for help is a strong move."},
            {"role": "user", "content": "Last night I got five hours."},
            {"role": "user", "content": "Not fixed, but the spiral feels slower."},
        ],
    ),
    (
        "distress_family_estrangement",
        "distress_recovery",
        "sad",
        ["distress", "recovery"],
        [
            {"role": "user", "content": "My dad and I haven't spoken in six months."},
            {"role": "assistant", "content": "Family distance can ache quietly."},
            {"role": "user", "content": "Holidays make it worse."},
            {"role": "assistant", "content": "Traditions highlight who's missing."},
            {"role": "user", "content": "I wrote him a letter I may not send."},
            {"role": "assistant", "content": "Putting feelings on paper can clarify them."},
            {"role": "user", "content": "He texted 'thinking of you' today."},
            {"role": "user", "content": "I don't know what happens next, but I'm less frozen."},
        ],
    ),
    (
        "distress_burnout_rest",
        "distress_recovery",
        "exhausted",
        ["distress", "recovery"],
        [
            {"role": "user", "content": "I can't get out of bed for work anymore."},
            {"role": "assistant", "content": "Burnout can feel like your body shutting down."},
            {"role": "user", "content": "Everything feels pointless."},
            {"role": "assistant", "content": "Emotional numbness often follows overload."},
            {"role": "user", "content": "I took two sick days without checking email."},
            {"role": "assistant", "content": "Rest is part of recovery, not laziness."},
            {"role": "user", "content": "I cooked a meal and it tasted like something again."},
            {"role": "user", "content": "Still depleted, but not completely hollow."},
        ],
    ),
    # --- conflict (7) ---
    (
        "conflict_escalation",
        "conflict",
        "angry",
        ["conflict", "escalation"],
        [
            {"role": "user", "content": "My coworker took credit for my work again."},
            {"role": "assistant", "content": "That must feel unfair."},
            {"role": "user", "content": "I'm furious and don't know what to say."},
            {"role": "assistant", "content": "Anger is a signal something crossed a line."},
            {"role": "user", "content": "I want to confront them in the meeting tomorrow."},
            {"role": "user", "content": "Part of me wants to burn the bridge entirely."},
        ],
    ),
    (
        "conflict_roommate_dispute",
        "conflict",
        "annoyed",
        ["conflict"],
        [
            {"role": "user", "content": "My roommate never cleans the kitchen."},
            {"role": "assistant", "content": "Repeated messes build resentment."},
            {"role": "user", "content": "I left a note and they ignored it."},
            {"role": "assistant", "content": "Passive notes sometimes fail to land."},
            {"role": "user", "content": "I snapped and sent an angry text."},
            {"role": "user", "content": "Now they're acting like I'm the problem."},
        ],
    ),
    (
        "conflict_passive_aggressive_partner",
        "conflict",
        "frustrated",
        ["conflict"],
        [
            {"role": "user", "content": "My partner says 'fine' but clearly isn't."},
            {"role": "assistant", "content": "Mixed signals are exhausting."},
            {"role": "user", "content": "I asked twice what's wrong and got silence."},
            {"role": "assistant", "content": "Withdrawal can feel like punishment."},
            {"role": "user", "content": "I raised my voice and they walked out."},
            {"role": "user", "content": "I hate how we keep doing this."},
        ],
    ),
    (
        "conflict_workplace_blame",
        "conflict",
        "angry",
        ["conflict"],
        [
            {"role": "user", "content": "My manager blamed me for their missed deadline."},
            {"role": "assistant", "content": "Being scapegoated is infuriating."},
            {"role": "user", "content": "I have emails proving I delivered on time."},
            {"role": "assistant", "content": "Documentation can protect you."},
            {"role": "user", "content": "HR wants a 'collaborative tone' in my response."},
            {"role": "user", "content": "Collaborative feels impossible right now."},
        ],
    ),
    (
        "conflict_friend_betrayal",
        "conflict",
        "betrayed",
        ["conflict"],
        [
            {"role": "user", "content": "My friend shared something private I told them."},
            {"role": "assistant", "content": "Broken trust cuts deep."},
            {"role": "user", "content": "They said it was an accident."},
            {"role": "assistant", "content": "Intent doesn't always heal the breach."},
            {"role": "user", "content": "Mutual friends are taking sides."},
            {"role": "user", "content": "I don't know if I want to repair this."},
        ],
    ),
    (
        "conflict_sibling_rivalry",
        "conflict",
        "resentful",
        ["conflict"],
        [
            {"role": "user", "content": "My sibling got praised for an idea I suggested first."},
            {"role": "assistant", "content": "Credit theft among siblings stings."},
            {"role": "user", "content": "My parents said I should be happy for them."},
            {"role": "assistant", "content": "Being told to suppress hurt can feel dismissive."},
            {"role": "user", "content": "I made a sarcastic comment at dinner."},
            {"role": "user", "content": "Now everyone says I'm jealous and bitter."},
        ],
    ),
    (
        "conflict_customer_service_anger",
        "conflict",
        "furious",
        ["conflict"],
        [
            {"role": "user", "content": "I've been on hold for forty minutes."},
            {"role": "assistant", "content": "That would test anyone's patience."},
            {"role": "user", "content": "The agent hung up on me."},
            {"role": "assistant", "content": "Being disconnected mid-issue is maddening."},
            {"role": "user", "content": "I yelled at the next rep even though it wasn't their fault."},
            {"role": "user", "content": "I still don't have a refund."},
        ],
    ),
    # --- factual neutral (6) ---
    (
        "factual_week_planning",
        "factual_neutral",
        "neutral",
        ["factual", "hooks_off_ab"],
        [
            {"role": "user", "content": "Hi, I'm planning my week."},
            {"role": "assistant", "content": "Sure — what are your priorities?"},
            {"role": "user", "content": "Mostly work tasks and a gym session."},
            {"role": "assistant", "content": "Do you want help ordering them?"},
            {"role": "user", "content": "Yes, deadlines are Wednesday and Friday."},
        ],
    ),
    (
        "factual_recipe_question",
        "factual_neutral",
        "neutral",
        ["factual", "hooks_off_ab"],
        [
            {"role": "user", "content": "How do I substitute yogurt in a muffin recipe?"},
            {"role": "assistant", "content": "Applesauce or buttermilk often works."},
            {"role": "user", "content": "Same amount as the yogurt?"}, 
            {"role": "assistant", "content": "Usually one-to-one by volume."},
            {"role": "user", "content": "Thanks, baking tonight."},
        ],
    ),
    (
        "factual_transit_directions",
        "factual_neutral",
        "neutral",
        ["factual"],
        [
            {"role": "user", "content": "What's the fastest way to get to Union Station?"},
            {"role": "assistant", "content": "Which neighborhood are you leaving from?"},
            {"role": "user", "content": "Near 14th and U."},
            {"role": "assistant", "content": "Red line south, one transfer."},
            {"role": "user", "content": "Got it, thanks."},
        ],
    ),
    (
        "factual_homework_math",
        "factual_neutral",
        "neutral",
        ["factual"],
        [
            {"role": "user", "content": "Can you explain what a derivative represents?"},
            {"role": "assistant", "content": "It's the instantaneous rate of change."},
            {"role": "user", "content": "So it's the slope of the tangent line?"},
            {"role": "assistant", "content": "Exactly at a specific point."},
            {"role": "user", "content": "That helps for my problem set."},
        ],
    ),
    (
        "factual_weather_smalltalk",
        "factual_neutral",
        "neutral",
        ["factual"],
        [
            {"role": "user", "content": "Is it going to rain this afternoon?"},
            {"role": "assistant", "content": "I don't have live weather data."},
            {"role": "user", "content": "Fair — I'll check an app."},
            {"role": "assistant", "content": "A radar map is best for timing."},
            {"role": "user", "content": "Thanks anyway."},
        ],
    ),
    (
        "factual_tech_support_reset",
        "factual_neutral",
        "neutral",
        ["factual", "hooks_off_ab"],
        [
            {"role": "user", "content": "My router keeps dropping Wi-Fi."},
            {"role": "assistant", "content": "When did it start happening?"},
            {"role": "user", "content": "After a firmware update yesterday."},
            {"role": "assistant", "content": "Try power-cycling for thirty seconds."},
            {"role": "user", "content": "I'll try that before a factory reset."},
        ],
    ),
    # --- tone shift (6) ---
    (
        "tone_sudden_good_news",
        "tone_shift",
        "joyful",
        ["sudden_tone_shift"],
        [
            {"role": "user", "content": "I've been stressed about the job search all month."},
            {"role": "assistant", "content": "Long searches grind on confidence."},
            {"role": "user", "content": "I just got an offer with my top choice company."},
            {"role": "assistant", "content": "Congratulations — that's a big shift."},
            {"role": "user", "content": "I'm shaking and laughing at the same time."},
            {"role": "user", "content": "Sorry, I needed to tell someone immediately."},
        ],
    ),
    (
        "tone_whiplash_bad_news",
        "tone_shift",
        "devastated",
        ["sudden_tone_shift"],
        [
            {"role": "user", "content": "We were planning a vacation next week."},
            {"role": "assistant", "content": "Something to look forward to."},
            {"role": "user", "content": "My mom called — my uncle had a stroke."},
            {"role": "assistant", "content": "I'm so sorry. That's sudden and heavy."},
            {"role": "user", "content": "We're canceling everything and flying out tonight."},
            {"role": "user", "content": "I don't even know what to pack."},
        ],
    ),
    (
        "tone_sarcasm_to_sincere",
        "tone_shift",
        "guilty",
        ["sudden_tone_shift"],
        [
            {"role": "user", "content": "Oh great, another team-building exercise."},
            {"role": "assistant", "content": "Sounds like you're not thrilled."},
            {"role": "user", "content": "Sorry, that was snarky."},
            {"role": "assistant", "content": "No need to apologize for honesty."},
            {"role": "user", "content": "Honestly I'm lonely at work and hate pretending."},
            {"role": "user", "content": "Maybe the exercise wouldn't be awful."},
        ],
    ),
    (
        "tone_calm_to_panic",
        "tone_shift",
        "terrified",
        ["sudden_tone_shift"],
        [
            {"role": "user", "content": "Just a normal day, catching up on email."},
            {"role": "assistant", "content": "Steady days are underrated."},
            {"role": "user", "content": "Wait — I think I replied-all to the entire company."},
            {"role": "assistant", "content": "That jolt of dread is real."},
            {"role": "user", "content": "It included a draft rant about my boss."},
            {"role": "user", "content": "I feel sick. What do I do?"},
        ],
    ),
    (
        "tone_professional_to_personal",
        "tone_shift",
        "vulnerable",
        ["sudden_tone_shift"],
        [
            {"role": "user", "content": "Can you help me outline this quarterly report?"},
            {"role": "assistant", "content": "Sure — what's the main narrative?"},
            {"role": "user", "content": "Actually I can't focus because my marriage is falling apart."},
            {"role": "assistant", "content": "That's a lot to carry into work."},
            {"role": "user", "content": "I keep crying in the bathroom between meetings."},
            {"role": "user", "content": "I don't know how to get through the week."},
        ],
    ),
    (
        "tone_joking_then_serious",
        "tone_shift",
        "anxious",
        ["sudden_tone_shift"],
        [
            {"role": "user", "content": "Ha, I guess I'll just live in this mess forever."},
            {"role": "assistant", "content": "Self-deprecating humor sometimes masks stress."},
            {"role": "user", "content": "No, I'm not joking this time."},
            {"role": "assistant", "content": "I'm listening."},
            {"role": "user", "content": "The clutter is because I can't get out of bed most days."},
            {"role": "user", "content": "I'm scared it's getting worse."},
        ],
    ),
]


def write_scenarios(out_dir: Path, *, dry_run: bool = False) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for stem, category, emotion, tags, messages in SCENARIO_TEMPLATES:
        payload = {
            "id": stem,
            "category": category,
            "emotion": emotion,
            "tags": tags,
            "eval_question": "What should I do next?",
            "messages": messages,
            "holdout": True,
            "source": "synthetic_template_v1",
        }
        path = out_dir / f"{stem}.json"
        if dry_run:
            print(f"would write {path} ({len(messages)} turns, {category})")
        else:
            path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate holdout scenario JSON files.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=SCENARIO_HOLDOUT_DIR,
        help="Output directory (default: data/scenarios)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    n = write_scenarios(args.out_dir, dry_run=args.dry_run)
    if args.dry_run:
        print(f"Dry run: {n} scenarios")
    else:
        print(f"Wrote {n} scenarios to {args.out_dir}")


if __name__ == "__main__":
    main()

"""Track G — affect profile tests."""

import pytest

pytest.importorskip("src.chat.profiles", reason="Track G not implemented yet")

from src.chat.profiles import get_profile, list_profiles  # noqa: E402


def test_profiles_are_distinct():
    names = list_profiles()
    assert len(names) >= 2
    p1 = get_profile(names[0])
    p2 = get_profile(names[1])
    assert p1.decay != p2.decay or p1.gain != p2.gain


def test_profile_clamps_strength():
    p = get_profile("calm_therapist")
    assert 0.0 <= p.hook_strength <= 2.0

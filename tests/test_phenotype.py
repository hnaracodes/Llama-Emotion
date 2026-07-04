"""Track C — phenotype report tests."""

from src.benchmark.phenotype import build_phenotype


def test_build_phenotype_empty():
    out = build_phenotype([])
    assert isinstance(out, dict)
    assert out.get("metric_type") == "heuristic"


def test_build_phenotype_has_signed_deltas():
    rows = [
        {
            "conditions": {
                "neutral": {"generated_preview": "Hello."},
                "high_affect": {"generated_preview": "I'm sorry you're anxious."},
            }
        }
    ]
    out = build_phenotype(rows)
    assert "empathy_delta" in out
    assert out["empathy_delta"] >= 0

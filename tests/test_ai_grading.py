from scanmyresume.services import ai_grading


def _base_report(score=60.0):
    return {
        "score": score,
        "components": {"required_coverage": 50.0},
        "matched_keywords": ["python"],
        "missing_keywords": ["docker"],
        "section_presence": {"experience": True},
    }


def test_redacts_direct_contact_data():
    text = "Jane jane@example.com +1 (555) 123-4567 https://example.com/jane"
    redacted = ai_grading.redact_contact_data(text)
    assert "jane@example.com" not in redacted
    assert "555" not in redacted
    assert "example.com" not in redacted
    assert "[EMAIL REDACTED]" in redacted


def test_ai_score_uses_fixed_dimension_weights():
    scores = {
        "role_alignment": 80,
        "evidence_impact": 60,
        "skills_coverage": 70,
        "clarity": 90,
        "ats_readability": 100,
    }
    assert ai_grading.calculate_ai_score(scores) == 76.0


def test_blend_cannot_move_grade_beyond_bound(monkeypatch):
    monkeypatch.setenv("AI_SCORE_WEIGHT", "0.5")
    monkeypatch.setenv("AI_MAX_ADJUSTMENT", "8")
    assert ai_grading.blend_score(60.0, 100.0) == (68.0, 8.0)
    assert ai_grading.blend_score(60.0, 0.0) == (52.0, -8.0)


def test_applies_structured_feedback_without_network(monkeypatch):
    monkeypatch.setenv("AI_GRADING_ON", "1")
    monkeypatch.setenv("AI_SCORE_WEIGHT", "0.25")
    monkeypatch.setenv("AI_MAX_ADJUSTMENT", "10")
    monkeypatch.setattr(
        ai_grading,
        "_call_groq",
        lambda *_: {
            "dimension_scores": {
                "role_alignment": 80,
                "evidence_impact": 80,
                "skills_coverage": 80,
                "clarity": 80,
                "ats_readability": 80,
            },
            "summary": "Relevant, but impact evidence can improve.",
            "strengths": ["Relevant Python experience."],
            "improvements": ["Quantify the project outcome."],
            "rewritten_bullets": ["Built Python tooling that improved [KPI] by [X%]."],
        },
    )

    result = ai_grading.apply_ai_grading(
        _base_report(), "resume", "job", requested=True
    )
    assert result["rule_based_score"] == 60.0
    assert result["score"] == 65.0
    assert result["ai_feedback"]["used"] is True
    assert result["ai_feedback"]["ai_quality_score"] == 80.0


def test_provider_failure_preserves_rule_score(monkeypatch):
    monkeypatch.setenv("AI_GRADING_ON", "1")

    def fail(*_):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(ai_grading, "_call_groq", fail)
    result = ai_grading.apply_ai_grading(
        _base_report(72.0), "resume", "job", requested=True
    )
    assert result["score"] == 72.0
    assert result["ai_feedback"]["status"] == "unavailable"

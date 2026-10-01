from pathlib import Path

from uacaa.cross_model import (
    VERIFIER_SCHEMA,
    evidence_cards,
    evaluate_cross_model_predictions,
    read_jsonl_cache,
    safe_surface_equivalent,
    write_cross_model_predictions,
    write_secondary_cache,
)
from uacaa.schemas import Context, EvidenceObservation, PreparedExample


def _example() -> PreparedExample:
    return PreparedExample(
        question_id="q1",
        split="dev",
        question="What was the capital in 2020?",
        contexts=[
            Context(0, "In 2020 Paris was the capital.", "agency.gov"),
            Context(1, "In 2019 Lyon was the capital.", "example.com"),
        ],
        supervision={
            "correct_answer": "Paris",
            "candidates": [
                {"answer": "Paris", "context_indices": [0]},
                {"answer": "Lyon", "context_indices": [1]},
            ],
        },
    )


def _primary():
    return {
        "answer": "Lyon",
        "evidence": [{"context_index": 1, "quote": "In 2019 Lyon was the capital"}],
        "alternatives": [],
        "contradictions": [],
        "evidence_sufficient": True,
        "confidence": "medium",
    }


def _secondary():
    return {
        "answer": "Paris",
        "evidence": [{"context_index": 0, "quote": "In 2020 Paris was the capital"}],
        "alternatives": [],
        "contradictions": [],
        "evidence_sufficient": True,
        "confidence": "high",
    }


class _SecondaryGenerator:
    def generate(self, example, primary, observations):
        return {
            "triggered": True,
            "risk_mode": "union",
            "conflict_probability": 0.8,
            "conflict_threshold": 0.2,
            "detector_trigger": True,
            "rule_trigger_reasons": [],
            "secondary_generation": _secondary(),
            "response_metadata": {"usage": {"total_tokens": 10}},
        }


class _Verifier:
    def verify(self, example, cards):
        assert [item["answer"] for item in cards] == ["Lyon", "Paris"]
        return (
            {
                "selected_candidate_id": "C2",
                "confidence": "high",
                "evidence_sufficient": True,
                "decisive_context_indices": [0],
                "counterevidence_context_indices": [1],
            },
            {"usage": {"total_tokens": 8}},
        )


def test_evidence_cards_are_source_time_aware_and_candidate_locked():
    # OpenAI Structured Outputs rejects JSON Schema's uniqueItems keyword;
    # verifier parsing performs deterministic de-duplication instead.
    assert "uniqueItems" not in VERIFIER_SCHEMA["properties"]["decisive_context_indices"]
    cards = evidence_cards(
        _example(),
        _primary(),
        _secondary(),
        [
            EvidenceObservation(0, "Paris", entailment=1, temporal_fit=1),
            EvidenceObservation(1, "Lyon", entailment=1, temporal_fit=0),
        ],
        {"agency.gov": {"authority": 0.9, "category": "official"}},
    )
    assert [item["answer"] for item in cards] == ["Lyon", "Paris"]
    assert cards[1]["question_time_match"] == 1.0
    assert cards[1]["evidence"][0]["source_category"] == "official"
    assert cards[1]["evidence"][0]["authority"] == 0.9


def test_cross_model_cache_switch_and_evaluation(tmp_path: Path):
    example = _example()
    primary = {"q1": {"generation": _primary()}}
    secondary_path = tmp_path / "secondary.jsonl"
    progress = write_secondary_cache(
        [example], primary, {}, _SecondaryGenerator(), secondary_path
    )
    assert progress["triggered"] == 1
    assert progress["llm_calls"] == 1
    predictions_path = tmp_path / "predictions.jsonl"
    result = write_cross_model_predictions(
        [example],
        primary,
        read_jsonl_cache(secondary_path),
        {},
        _Verifier(),
        predictions_path,
        mode="conservative",
    )
    assert result["switches"] == 1
    predictions = read_jsonl_cache(predictions_path)
    assert predictions["q1"]["initial_answer"] == "Lyon"
    assert predictions["q1"]["final_answer"] == "Paris"
    summary = evaluate_cross_model_predictions([example], predictions)
    assert summary["routing"]["helped"] == 1
    assert summary["routing"]["hurt"] == 0
    assert summary["cross_model_final"]["answer_exact"] == 1.0


def test_conservative_mode_rejects_unsubstantiated_switch(tmp_path: Path):
    class LowVerifier(_Verifier):
        def verify(self, example, cards):
            value, metadata = super().verify(example, cards)
            value["confidence"] = "low"
            return value, metadata

    example = _example()
    output = tmp_path / "predictions.jsonl"
    write_cross_model_predictions(
        [example],
        {"q1": {"generation": _primary()}},
        {"q1": {"triggered": True, "secondary_generation": _secondary()}},
        {},
        LowVerifier(),
        output,
        mode="conservative",
    )
    assert read_jsonl_cache(output)["q1"]["final_answer"] == "Lyon"


def test_safe_surface_equivalence_is_narrow():
    assert safe_surface_equivalent("October", "Every October.")
    assert safe_surface_equivalent("October", "Throughout the month of October")
    assert not safe_surface_equivalent("1994", "October 31, 1994")
    assert not safe_surface_equivalent("Paris", "Paris, Texas")
    assert not safe_surface_equivalent("30", "38")


def test_cached_equivalent_switch_is_reconciled_without_verifier_call(tmp_path: Path):
    class NoCallVerifier:
        def verify(self, example, cards):
            raise AssertionError("a complete cached record must not call the verifier")

    example = _example()
    output = tmp_path / "predictions.jsonl"
    output.write_text(
        '{"question_id":"q1","initial_answer":"October",'
        '"final_answer":"Every October.","triggered":true,'
        '"decision":"verifier_switch","verification":{},"evidence_cards":[]}\n',
        encoding="utf-8",
    )
    progress = write_cross_model_predictions(
        [example], {}, {}, {}, NoCallVerifier(), output, mode="conservative"
    )
    prediction = read_jsonl_cache(output)["q1"]
    assert progress["extracted"] == 0
    assert progress["reconciled_equivalent_switches"] == 1
    assert progress["switches"] == 0
    assert prediction["final_answer"] == "October"
    assert prediction["decision"] == "verifier_equivalent_keep_initial"


def test_cached_verification_rejects_changed_secondary_record(tmp_path: Path):
    example = _example()
    primary = {"q1": {"generation": _primary()}}
    original_secondary = {
        "q1": {"triggered": True, "secondary_generation": _secondary()}
    }
    output = tmp_path / "predictions.jsonl"
    write_cross_model_predictions(
        [example], primary, original_secondary, {}, _Verifier(), output,
        mode="conservative",
    )
    changed = _secondary()
    changed["answer"] = "Marseille"
    changed_secondary = {
        "q1": {"triggered": True, "secondary_generation": changed}
    }
    try:
        write_cross_model_predictions(
            [example], primary, changed_secondary, {}, _Verifier(), output,
            mode="conservative",
        )
    except ValueError as error:
        assert "changed secondary record: q1" in str(error)
    else:
        raise AssertionError("changed secondary source must invalidate cached verification")

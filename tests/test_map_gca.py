from pathlib import Path

from uacaa.map_gca import (
    BayesianRelationMAP,
    MAPGuidedResolver,
    evaluate_map_resolutions,
    generation_route,
    relation_matrix,
)
from uacaa.schemas import Context, PreparedExample


def _example(question_id: str, gold: str = "Paris") -> PreparedExample:
    return PreparedExample(
        question_id=question_id,
        split="train",
        question="What is the capital?",
        contexts=[
            Context(0, "Paris is the capital.", "a.example"),
            Context(1, "Lyon is the capital.", "b.example"),
            Context(2, "The capital is Paris.", "c.example"),
        ],
        supervision={"correct_answer": gold},
    )


def _conflict_generation():
    return {
        "answer": "Lyon",
        "evidence": [{"context_index": 1, "quote": "Lyon is the capital"}],
        "alternatives": [
            {
                "answer": "Paris",
                "evidence": [
                    {"context_index": 0, "quote": "Paris is the capital"},
                    {"context_index": 2, "quote": "The capital is Paris"},
                ],
            }
        ],
        "contradictions": [],
        "evidence_sufficient": True,
        "confidence": "medium",
        "citation_validation": {"valid": 3, "total": 3},
    }


def test_routing_distinguishes_direct_insufficient_and_conflict():
    example = _example("q1")
    conflict = _conflict_generation()
    assert generation_route(conflict, example) == "conflict"
    direct = {
        **conflict,
        "answer": "Paris",
        "evidence": conflict["alternatives"][0]["evidence"],
        "alternatives": [],
    }
    assert generation_route(direct, example) == "direct"
    assert generation_route({**direct, "evidence_sufficient": False}, example) == "insufficient"
    matrix = relation_matrix(conflict, example)
    assert [item["relation"] for item in matrix[0]["relations"]] == [
        "contradict",
        "support",
        "contradict",
    ]


def test_dirichlet_relation_map_fits_saves_and_selects_supported_gold(tmp_path: Path):
    examples = [_example(f"q{index}") for index in range(5)]
    generations = {
        example.question_id: {"generation": _conflict_generation()} for example in examples
    }
    model = BayesianRelationMAP(dependency_correction=False, accept_probability=0.6)
    report = model.fit(examples, generations)
    assert report["questions"] == 5
    posterior = model.posterior(_conflict_generation(), examples[0])
    assert posterior["map_answer"] == "Paris"
    assert posterior["max_probability"] > 0.5
    path = tmp_path / "map.json"
    model.save(path)
    loaded = BayesianRelationMAP.load(path)
    assert loaded.posterior(_conflict_generation(), examples[0])["map_answer"] == "Paris"


def test_map_only_resolution_improves_wrong_initial_answer():
    example = _example("q1")
    model = BayesianRelationMAP(dependency_correction=False)
    model.fit([example], {"q1": {"generation": _conflict_generation()}})
    resolver = MAPGuidedResolver(model, "unused", mode="map-only")
    record = resolver.resolve(example, {"generation": _conflict_generation()})
    assert record["route"] == "conflict"
    assert record["decision"] == "map_direct"
    assert record["initial_answer"] == "Lyon"
    assert record["final_answer"] == "Paris"
    summary = evaluate_map_resolutions(
        [example],
        {"q1": {"question_id": "q1", **record}},
    )
    assert summary["adaptive_final"]["answer_exact"] == 1.0
    assert summary["routing"]["helped"] == 1
    assert summary["conflict_only"]["final_exact"] == 1.0

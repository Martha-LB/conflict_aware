from uacaa.normalization import (
    answers_match,
    best_answer_index,
    exact_match,
    normalize_answer,
    qacc_exact_match,
    qacc_normalize_text,
    qacc_token_f1,
    safe_surface_equivalent,
    target_matches_candidate,
    token_f1,
)


def test_qacc_metrics_reproduce_released_evaluator():
    assert qacc_normalize_text("The U.S.A.") == "usa"
    assert qacc_exact_match("The Beatles", "Beatles")
    assert not qacc_exact_match("Yamuna", "Yamuna River")
    assert not qacc_exact_match("3:2", "2:3")
    assert qacc_normalize_text("3:2") == "32"
    assert qacc_normalize_text("2:3") == "23"
    assert qacc_token_f1("Season 5 Episode 17", "Season 5") == 2 * (2 / 4) / (1 + 2 / 4)


def test_normalization_handles_articles_and_number_commas():
    assert normalize_answer("The 5,000") == "5000"
    assert answers_match("The Eiffel Tower", "Eiffel Tower")
    assert exact_match("The Eiffel Tower", "Eiffel Tower")
    assert not exact_match("Eiffel Tower", "Eiffel Tower in Paris")


def test_numeric_conflicts_do_not_alias():
    assert not answers_match("1850", "1858")
    assert not answers_match("3:2", "2:3")
    assert not target_matches_candidate("3:2", "2:3")


def test_safe_surface_equivalence_preserves_material_details():
    assert safe_surface_equivalent("October", "Throughout the month of October")
    assert not safe_surface_equivalent("1994", "October 31, 1994")
    assert not safe_surface_equivalent("Paris", "Paris, Texas")


def test_candidate_mapping_allows_safe_gold_containment():
    assert target_matches_candidate("16 years", "16 years of age")
    assert target_matches_candidate(
        "Western Eastern Ghats",
        "the Western Ghats and the Eastern Ghats",
    )
    assert not target_matches_candidate("16 years", "16 years old in 2028")
    assert not answers_match("16 years", "16 years of age")


def test_candidate_mapping_handles_orthographic_and_count_variants():
    assert target_matches_candidate("meristematic tissue", "meristematic tissues")
    assert target_matches_candidate("Jimmy Mathews", "Jimmy Matthews")
    assert target_matches_candidate("June 28, 2018", "Jun 28, 2018")
    assert target_matches_candidate("10 episodes", "10")
    assert target_matches_candidate("two seats", "2")
    assert target_matches_candidate("$160,000.00", "$160,000")
    assert not target_matches_candidate("12 episodes", "13")


def test_best_answer_index_uses_alias_similarity():
    assert best_answer_index("Fourteenth Amendment", ["First Amendment", "The Fourteenth Amendment"]) == 1
    assert token_f1("December 1972", "December 7 19 1972") > 0

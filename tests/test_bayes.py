from pathlib import Path
from types import MethodType

import numpy as np

from uacaa.bayes import BayesianConditionalLogit
from uacaa.schemas import ChoiceSet


def _sets(count=20):
    result = []
    for index in range(count):
        result.append(
            ChoiceSet(
                question_id=str(index),
                answers=("wrong", "right"),
                features=np.asarray([[0.0, 1.0], [1.0, 0.0]]),
                correct_index=1,
            )
        )
    return result


def test_bayesian_conditional_logit_fits_saves_and_loads(tmp_path: Path):
    model = BayesianConditionalLogit(("signal", "noise"), posterior_samples=200, seed=7)
    summary = model.fit(_sets())
    assert summary.converged
    model.prediction_method = "map"
    model.dependency_correction = False
    model.calibrate_temperature(_sets(5))
    model.select_abstention_threshold(_sets(5), minimum_coverage=1.0)
    probabilities = model.posterior_probabilities(_sets(1)[0])
    assert probabilities[1] > probabilities[0]
    path = tmp_path / "model.json"
    model.save(path)
    loaded = BayesianConditionalLogit.load(path)
    assert loaded.map_probabilities(_sets(1)[0])[1] > 0.5
    assert loaded.prediction_method == "map"
    assert loaded.dependency_correction is False


def test_abstention_threshold_ignores_single_candidate_rows():
    model = BayesianConditionalLogit(("signal", "noise"), posterior_samples=10)
    model.fit(_sets())
    competing = [
        ChoiceSet(
            question_id="correct-low",
            answers=("wrong", "right"),
            features=np.asarray([[0.0, 1.0], [1.0, 0.0]]),
            correct_index=1,
        ),
        ChoiceSet(
            question_id="wrong-high",
            answers=("right", "wrong"),
            features=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
            correct_index=0,
        ),
    ]
    singles = [
        ChoiceSet(
            question_id=f"single-{index}",
            answers=("right",),
            features=np.asarray([[1.0, 0.0]]),
            correct_index=0,
        )
        for index in range(10)
    ]

    def fixed_probabilities(self, item):
        if item.question_id == "correct-low":
            return np.asarray([0.4, 0.6])
        if item.question_id == "wrong-high":
            return np.asarray([0.1, 0.9])
        return np.ones(1)

    model.posterior_probabilities = MethodType(fixed_probabilities, model)
    threshold = model.select_abstention_threshold(
        [*competing, *singles], minimum_coverage=0.5
    )
    assert threshold == 0.6

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp, softmax

from .schemas import ChoiceSet, Prediction


@dataclass
class FitSummary:
    questions: int
    candidates: int
    converged: bool
    objective: float


class BayesianConditionalLogit:
    """Conditional-logit choice model with Gaussian priors and Laplace posterior.

    The model operates over candidate answers within a question. Feature
    standardization is learned on the training split. The inverse Hessian at the
    MAP estimate approximates parameter uncertainty, and posterior predictive
    probabilities are Monte-Carlo averages over that Gaussian approximation.
    """

    def __init__(
        self,
        feature_names: Sequence[str],
        *,
        prior_scale: float = 2.0,
        posterior_samples: int = 1000,
        label_smoothing: float = 0.05,
        seed: int = 42,
    ) -> None:
        if prior_scale <= 0:
            raise ValueError("prior_scale must be positive")
        if posterior_samples < 1:
            raise ValueError("posterior_samples must be positive")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        self.feature_names = tuple(feature_names)
        self.prior_scale = float(prior_scale)
        self.posterior_samples = int(posterior_samples)
        self.label_smoothing = float(label_smoothing)
        self.seed = int(seed)
        self.feature_mean: np.ndarray | None = None
        self.feature_scale: np.ndarray | None = None
        self.map_coefficients: np.ndarray | None = None
        self.covariance: np.ndarray | None = None
        self.temperature = 1.0
        self.abstention_threshold = 0.55
        self.prediction_method = "posterior"
        self.dependency_correction = True

    def _check_width(self, matrix: np.ndarray) -> None:
        if matrix.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features, got {matrix.shape[1]}"
            )

    def _transform(self, matrix: np.ndarray) -> np.ndarray:
        if self.feature_mean is None or self.feature_scale is None:
            raise RuntimeError("Model is not fitted")
        self._check_width(matrix)
        return (matrix - self.feature_mean) / self.feature_scale

    def fit(self, choice_sets: Sequence[ChoiceSet]) -> FitSummary:
        usable = [item for item in choice_sets if item.correct_index is not None and len(item.answers) >= 2]
        if not usable:
            raise ValueError("No labelled multi-candidate choice sets were supplied")
        all_features = np.vstack([item.features for item in usable])
        self._check_width(all_features)
        self.feature_mean = all_features.mean(axis=0)
        raw_scale = all_features.std(axis=0)
        self.feature_scale = np.where(raw_scale > 1e-8, raw_scale, 1.0)
        transformed = [self._transform(item.features) for item in usable]
        dimension = len(self.feature_names)
        prior_precision = 1.0 / (self.prior_scale**2)

        def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
            loss = 0.5 * prior_precision * float(beta @ beta)
            gradient = prior_precision * beta.copy()
            for item, matrix in zip(usable, transformed):
                scores = matrix @ beta
                probabilities = softmax(scores)
                size = len(item.answers)
                target = np.full(size, self.label_smoothing / size)
                target[int(item.correct_index)] += 1.0 - self.label_smoothing
                loss += float(logsumexp(scores) - target @ scores)
                gradient += matrix.T @ (probabilities - target)
            return loss, gradient

        result = minimize(
            lambda beta: objective(beta),
            np.zeros(dimension, dtype=float),
            jac=True,
            method="L-BFGS-B",
        )
        self.map_coefficients = np.asarray(result.x, dtype=float)

        hessian = np.eye(dimension) * prior_precision
        for matrix in transformed:
            probabilities = softmax(matrix @ self.map_coefficients)
            weight = np.diag(probabilities) - np.outer(probabilities, probabilities)
            hessian += matrix.T @ weight @ matrix
        covariance = np.linalg.pinv(hessian)
        covariance = (covariance + covariance.T) / 2.0
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        self.covariance = eigenvectors @ np.diag(np.clip(eigenvalues, 1e-10, None)) @ eigenvectors.T
        return FitSummary(
            questions=len(usable),
            candidates=sum(len(item.answers) for item in usable),
            converged=bool(result.success),
            objective=float(result.fun),
        )

    def _require_fit(self) -> tuple[np.ndarray, np.ndarray]:
        if self.map_coefficients is None or self.covariance is None:
            raise RuntimeError("Model is not fitted")
        return self.map_coefficients, self.covariance

    def map_probabilities(self, choice_set: ChoiceSet) -> np.ndarray:
        coefficients, _ = self._require_fit()
        scores = self._transform(choice_set.features) @ coefficients
        return softmax(scores / self.temperature)

    def posterior_probabilities(self, choice_set: ChoiceSet) -> np.ndarray:
        coefficients, covariance = self._require_fit()
        matrix = self._transform(choice_set.features)
        if len(choice_set.answers) == 1:
            return np.ones(1, dtype=float)
        rng = np.random.default_rng(self.seed)
        samples = rng.multivariate_normal(
            coefficients,
            covariance,
            size=self.posterior_samples,
            check_valid="ignore",
        )
        scores = samples @ matrix.T / self.temperature
        return softmax(scores, axis=1).mean(axis=0)

    def probabilities(self, choice_set: ChoiceSet) -> np.ndarray:
        if self.prediction_method == "posterior":
            return self.posterior_probabilities(choice_set)
        if self.prediction_method == "map":
            return self.map_probabilities(choice_set)
        raise ValueError("prediction_method must be 'posterior' or 'map'")

    def calibrate_temperature(
        self,
        choice_sets: Sequence[ChoiceSet],
        *,
        probability_method: str = "posterior",
    ) -> float:
        usable = [item for item in choice_sets if item.correct_index is not None and len(item.answers) >= 2]
        if not usable:
            return self.temperature
        coefficients, covariance = self._require_fit()
        transformed = [self._transform(item.features) for item in usable]
        posterior_scores: list[np.ndarray] | None = None
        if probability_method == "posterior":
            rng = np.random.default_rng(self.seed)
            samples = rng.multivariate_normal(
                coefficients,
                covariance,
                size=self.posterior_samples,
                check_valid="ignore",
            )
            posterior_scores = [samples @ matrix.T for matrix in transformed]
        elif probability_method != "map":
            raise ValueError("probability_method must be 'posterior' or 'map'")

        def nll(log_temperature: float) -> float:
            temperature = float(np.exp(log_temperature))
            loss = 0.0
            for index, (item, matrix) in enumerate(zip(usable, transformed)):
                if posterior_scores is None:
                    scores = matrix @ coefficients / temperature
                    loss += float(logsumexp(scores) - scores[int(item.correct_index)])
                else:
                    probabilities = softmax(
                        posterior_scores[index] / temperature,
                        axis=1,
                    ).mean(axis=0)
                    loss -= float(np.log(max(probabilities[int(item.correct_index)], 1e-12)))
            return loss / len(usable)

        result = minimize_scalar(
            nll,
            bounds=(np.log(0.25), np.log(4.0)),
            method="bounded",
        )
        self.temperature = float(np.exp(result.x))
        return self.temperature

    def select_abstention_threshold(
        self,
        choice_sets: Sequence[ChoiceSet],
        *,
        minimum_coverage: float = 0.7,
        probability_method: str = "posterior",
    ) -> float:
        # A one-candidate softmax is identically one, regardless of evidence
        # quality or parameter uncertainty.  Such rows are useful for answer
        # evaluation, but cannot identify a selective-prediction threshold.
        # Calibrate coverage only on questions where the ranker actually has
        # to resolve competing candidates.
        labelled = [
            item
            for item in choice_sets
            if item.correct_index is not None and len(item.answers) >= 2
        ]
        if not labelled:
            return self.abstention_threshold
        rows = []
        for item in labelled:
            if probability_method == "posterior":
                probabilities = self.posterior_probabilities(item)
            elif probability_method == "map":
                probabilities = self.map_probabilities(item)
            else:
                raise ValueError("probability_method must be 'posterior' or 'map'")
            prediction = int(np.argmax(probabilities))
            rows.append((float(probabilities[prediction]), prediction == item.correct_index))
        candidates = sorted({0.0, *(confidence for confidence, _ in rows)})
        viable: list[tuple[float, float, float]] = []
        for threshold in candidates:
            selected = [correct for confidence, correct in rows if confidence >= threshold]
            coverage = len(selected) / len(rows)
            if selected and coverage >= minimum_coverage:
                viable.append((sum(selected) / len(selected), coverage, threshold))
        if viable:
            _, _, self.abstention_threshold = max(viable, key=lambda row: (row[0], row[1], row[2]))
        return self.abstention_threshold

    def predict(self, choice_set: ChoiceSet) -> Prediction:
        if not choice_set.answers:
            return Prediction(choice_set.question_id, None, (), 0.0, True)
        probabilities = self.probabilities(choice_set)
        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])
        abstained = confidence < self.abstention_threshold
        return Prediction(
            question_id=choice_set.question_id,
            answer=None if abstained else choice_set.answers[index],
            probabilities=tuple(float(value) for value in probabilities),
            confidence=confidence,
            abstained=abstained,
        )

    def save(self, path: Path) -> None:
        coefficients, covariance = self._require_fit()
        assert self.feature_mean is not None and self.feature_scale is not None
        payload = {
            "model_type": "bayesian_conditional_logit_laplace",
            "feature_names": self.feature_names,
            "prior_scale": self.prior_scale,
            "posterior_samples": self.posterior_samples,
            "label_smoothing": self.label_smoothing,
            "seed": self.seed,
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "map_coefficients": coefficients.tolist(),
            "covariance": covariance.tolist(),
            "temperature": self.temperature,
            "abstention_threshold": self.abstention_threshold,
            "prediction_method": self.prediction_method,
            "dependency_correction": self.dependency_correction,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BayesianConditionalLogit":
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = cls(
            payload["feature_names"],
            prior_scale=payload["prior_scale"],
            posterior_samples=payload["posterior_samples"],
            label_smoothing=payload["label_smoothing"],
            seed=payload["seed"],
        )
        model.feature_mean = np.asarray(payload["feature_mean"], dtype=float)
        model.feature_scale = np.asarray(payload["feature_scale"], dtype=float)
        model.map_coefficients = np.asarray(payload["map_coefficients"], dtype=float)
        model.covariance = np.asarray(payload["covariance"], dtype=float)
        model.temperature = float(payload["temperature"])
        model.abstention_threshold = float(payload["abstention_threshold"])
        model.prediction_method = str(payload.get("prediction_method", "posterior"))
        model.dependency_correction = bool(payload.get("dependency_correction", True))
        return model

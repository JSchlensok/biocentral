"""Engineering iterations can score caller-provided candidates instead of generated mutations."""

from types import SimpleNamespace

import pytest
from biotrainer_core.data_classes import SequenceData
from pydantic import ValidationError

from biocentral_server.active_learning.al_config import (
    ActiveLearningEngineeringCampaignConfig,
    ActiveLearningEngineeringIterationConfig,
    ActiveLearningModelType,
    ActiveLearningOptimizationMode,
)
from biocentral_server.active_learning.endpoint_models import (
    ActiveLearningEngineeringIterationRequest,
)
from biocentral_server.active_learning.pipelines.protein_engineering.steps.mutation_generation_step import (
    MutationGenerationStep,
)

WILDTYPE = "ACDEFGHIKL"


def _iteration_config(**overrides) -> ActiveLearningEngineeringIterationConfig:
    return ActiveLearningEngineeringIterationConfig(
        **{
            "iteration": 1,
            "training_data": [SequenceData(seq_id="wt", seq=WILDTYPE, label="1.0")],
            "coefficient": 0.5,
            "n_suggestions": 1,
            **overrides,
        }
    )


def _request(iteration_config) -> ActiveLearningEngineeringIterationRequest:
    return ActiveLearningEngineeringIterationRequest(
        campaign_config=ActiveLearningEngineeringCampaignConfig(
            name="candidates",
            model_type=ActiveLearningModelType.GAUSSIAN_PROCESS,
            optimization_mode=ActiveLearningOptimizationMode.MAXIMIZE,
            embedder_name="one_hot_encoding",
            wildtype_sequence=WILDTYPE,
        ),
        iteration_config=iteration_config,
    )


class TestConfig:
    def test_candidates_default_to_none(self):
        assert _iteration_config().candidate_sequences is None

    def test_rejects_duplicate_candidates(self):
        with pytest.raises(ValidationError, match="duplicate"):
            _iteration_config(candidate_sequences=["ACDEFGHIKM", "ACDEFGHIKM"])

    def test_rejects_candidates_of_a_different_length(self):
        with pytest.raises(ValidationError, match="same length"):
            _request(_iteration_config(candidate_sequences=["ACDEFGHIK"]))

    def test_accepts_candidates_matching_the_wildtype_length(self):
        request = _request(_iteration_config(candidate_sequences=["ACDEFGHIKM"]))
        assert request.iteration_config.candidate_sequences == ["ACDEFGHIKM"]


class TestMutationGenerationStep:
    def test_given_candidates_are_scored_unchanged(self):
        candidates = ["ACDEFGHIKM", "MCDEFGHIKL"]
        context = SimpleNamespace(
            base_sequences=[WILDTYPE],
            candidate_sequences=candidates,
            al_training_data=[SequenceData(seq_id="wt", seq=WILDTYPE, label="1.0")],
            n_mutations=1000,
            mutations=None,
        )

        result = MutationGenerationStep()._execute(context)

        assert result.mutations == candidates

    def test_mutations_are_generated_without_candidates(self):
        context = SimpleNamespace(
            base_sequences=[WILDTYPE],
            candidate_sequences=None,
            al_training_data=[SequenceData(seq_id="wt", seq=WILDTYPE, label="1.0")],
            n_mutations=5,
            mutations=None,
        )

        result = MutationGenerationStep()._execute(context)

        assert len(result.mutations) == 5
        assert WILDTYPE not in result.mutations

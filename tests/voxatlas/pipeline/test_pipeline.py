import pandas as pd

from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import ScalarFeatureOutput
from voxatlas.pipeline.feature_store import FeatureStore
from voxatlas.registry.feature_registry import registry
from voxatlas.pipeline.pipeline import VoxAtlasPipeline


class DummyFeature(BaseExtractor):
    name = "test.pipeline.feature"
    output_units = "frame"
    dependencies = []
    compute_calls = 0
    default_config = {
        "frame_step": 0.02,
    }
    last_params = None

    def compute(self, feature_input, params):
        type(self).compute_calls += 1
        type(self).last_params = params
        frames = feature_input.units.frames

        values = pd.Series([1] * len(frames))

        return ScalarFeatureOutput(
            feature=self.name,
            unit="frame",
            values=values,
        )


registry.register(DummyFeature)


def test_pipeline_run(dummy_audio, dummy_units):
    DummyFeature.compute_calls = 0
    DummyFeature.last_params = None

    config = {
        "features": ["test.pipeline.feature"],
        "feature_config": {
            "test.pipeline.feature": {
                "frame_step": 0.01,
            },
        },
        "pipeline": {
            "cache": False,
        },
    }

    pipeline = VoxAtlasPipeline(
        dummy_audio,
        dummy_units,
        config,
    )

    results = pipeline.run()

    assert isinstance(results, FeatureStore)
    assert results.exists("test.pipeline.feature")
    assert DummyFeature.last_params == {
        "frame_step": 0.01,
    }


def test_pipeline_uses_disk_cache(tmp_path, dummy_audio, dummy_units):
    DummyFeature.compute_calls = 0
    DummyFeature.last_params = None

    config = {
        "features": ["test.pipeline.feature"],
        "pipeline": {
            "cache": True,
            "cache_dir": str(tmp_path / ".voxatlas_cache"),
        },
    }

    first_pipeline = VoxAtlasPipeline(
        dummy_audio,
        dummy_units,
        config,
    )
    second_pipeline = VoxAtlasPipeline(
        dummy_audio,
        dummy_units,
        config,
    )

    first_results = first_pipeline.run()
    second_results = second_pipeline.run()

    assert first_results.exists("test.pipeline.feature")
    assert second_results.exists("test.pipeline.feature")
    assert DummyFeature.compute_calls == 1

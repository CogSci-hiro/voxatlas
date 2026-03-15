import hashlib
import json

from voxatlas.config.feature_config import resolve_feature_config
from voxatlas.core.discovery import discover_features
from voxatlas.features.feature_input import FeatureInput
from voxatlas.pipeline.cache import DiskCache
from voxatlas.pipeline.execution_plan import ExecutionPlan
from voxatlas.pipeline.executor import parallel_execute_layer
from voxatlas.pipeline.feature_store import FeatureStore
from voxatlas.registry.feature_registry import registry


class VoxAtlasPipeline:
    """
    Run a VoxAtlas feature extraction workflow for a single stream.

    A pipeline instance combines one audio stream, one unit hierarchy, and a
    runtime configuration. It validates requested features, resolves dependency
    layers, executes extractors in order, and stores intermediate results so
    downstream features can reuse them.

    Parameters
    ----------
    audio : Audio | None
        Audio stream for the current conversation channel. Acoustic features
        require this input.
    units : Units | None
        Hierarchical unit container for the current stream. Linguistic and
        alignment-based features require this input.
    config : dict
        Runtime configuration containing the requested features and pipeline
        options.

    Returns
    -------
    VoxAtlasPipeline
        Configured pipeline instance ready to execute.

    Notes
    -----
    VoxAtlas resolves dependencies through the feature registry and executes
    each dependency layer sequentially while allowing optional parallelism
    inside a layer.

    Examples
    --------
    Usage example::

        from voxatlas.io import load_dataset
        from voxatlas.pipeline import Pipeline

        dataset = load_dataset("/path/to/dataset", "conversation01")
        stream = dataset.streams()[0]
        pipeline = Pipeline(
            audio=stream.audio,
            units=stream.units,
            config={
                "features": ["acoustic.pitch.f0"],
                "pipeline": {"n_jobs": 1, "cache": False},
            },
        )
        results = pipeline.run()
        print(results.get("acoustic.pitch.f0"))
    """

    def __init__(self, audio, units, config):
        self.audio = audio
        self.units = units
        self.config = config

    def _validate_features(self) -> None:
        discover_features()

        for feature_name in self.config["features"]:
            registry.get(feature_name)

    def _resolve_dependencies(self) -> dict[str, list[str]]:
        dependency_map = {}
        visiting = set()
        visited = set()

        def visit(feature_name: str) -> None:
            if feature_name in visited:
                return

            if feature_name in visiting:
                raise ValueError(
                    f"Circular dependency detected for feature '{feature_name}'"
                )

            visiting.add(feature_name)

            extractor_cls = registry.get(feature_name)
            dependencies = list(getattr(extractor_cls, "dependencies", []))

            for dependency_name in dependencies:
                registry.get(dependency_name)
                visit(dependency_name)

            visiting.remove(feature_name)
            visited.add(feature_name)
            dependency_map[feature_name] = dependencies

        for feature_name in self.config["features"]:
            visit(feature_name)

        return dependency_map

    def _build_execution_plan(self) -> ExecutionPlan:
        dependency_map = self._resolve_dependencies()
        levels = {}

        def resolve_level(feature_name: str) -> int:
            if feature_name in levels:
                return levels[feature_name]

            dependencies = dependency_map[feature_name]

            if not dependencies:
                levels[feature_name] = 0
            else:
                levels[feature_name] = (
                    max(resolve_level(dependency_name) for dependency_name in dependencies)
                    + 1
                )

            return levels[feature_name]

        for feature_name in dependency_map:
            resolve_level(feature_name)

        layers = []

        for feature_name in dependency_map:
            level = levels[feature_name]

            while len(layers) <= level:
                layers.append([])

            layers[level].append(feature_name)

        return ExecutionPlan(layers)

    def _build_feature_input(self, feature_store: FeatureStore) -> FeatureInput:
        return FeatureInput(
            audio=self.audio,
            units=self.units,
            context={
                "config": self.config,
                "feature_store": feature_store,
            },
        )

    def _audio_hash(self) -> str:
        if self.audio is None:
            return hashlib.sha256(b"no-audio").hexdigest()

        payload = self.audio.waveform.tobytes()
        payload += str(self.audio.sample_rate).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _config_hash(self) -> str:
        payload = json.dumps(self.config, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _build_cache(self):
        pipeline_config = self.config.get("pipeline", {})

        if not pipeline_config.get("cache", False):
            return None

        cache_dir = pipeline_config.get("cache_dir", ".voxatlas_cache")
        return DiskCache(cache_dir)

    def _n_jobs(self) -> int:
        pipeline_config = self.config.get("pipeline", {})
        return max(1, int(pipeline_config.get("n_jobs", 1)))

    def run(self) -> FeatureStore:
        """
        Execute the configured feature graph and return computed outputs.

        The pipeline validates the requested features, creates an execution
        plan from registry dependencies, then executes each dependency layer in
        order. Intermediate outputs are inserted into a feature store so later
        features can retrieve them.

        Returns
        -------
        FeatureStore
            Store containing requested features and any computed dependencies.

        Raises
        ------
        ValueError
            Raised when the dependency graph contains a cycle.
        KeyError
            Raised when a required feature is missing from the store or cache
            during execution.

        Notes
        -----
        When caching is enabled, cached outputs are loaded before an extractor
        is scheduled for execution.

        Examples
        --------
        Usage example::

            results = pipeline.run()
            print(results.exists("acoustic.pitch.f0"))
        """
        self._validate_features()
        execution_plan = self._build_execution_plan()
        feature_store = FeatureStore()
        feature_input = self._build_feature_input(feature_store)
        cache = self._build_cache()
        n_jobs = self._n_jobs()

        if cache is not None:
            audio_hash = self._audio_hash()
            config_hash = self._config_hash()

        for layer in execution_plan.layers:
            missing_features = []
            cache_keys = {}

            for feature_name in layer:
                if feature_store.exists(feature_name):
                    continue

                if cache is not None:
                    cache_key = cache.compute_key(feature_name, audio_hash, config_hash)
                    cache_keys[feature_name] = cache_key

                    if cache.exists(feature_name, cache_key):
                        result = cache.load(feature_name, cache_key)
                        feature_store.add(feature_name, result)
                        continue

                missing_features.append(feature_name)

            if not missing_features:
                continue

            feature_params = {
                feature_name: resolve_feature_config(
                    feature_name,
                    registry.get(feature_name),
                    self.config,
                )
                for feature_name in missing_features
            }

            layer_results = parallel_execute_layer(
                missing_features,
                registry,
                feature_input,
                n_jobs,
                feature_params,
            )

            for feature_name, result in layer_results.items():
                if cache is not None:
                    cache.save(feature_name, cache_keys[feature_name], result)

                feature_store.add(feature_name, result)

        return feature_store

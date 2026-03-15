import hashlib
import pickle
from pathlib import Path


class DiskCache:
    """
    Represent the disk cache concept in VoxAtlas.
    
    This public class exposes reusable state or behavior for the pipeline layer of VoxAtlas. It is part of the supported API surface and is intended to be composed by pipelines, registries, and feature extractors.
    
    Parameters
    ----------
    cache_dir : object
        Directory where serialized feature outputs are stored between pipeline runs.
    
    Examples
    --------
        from voxatlas.pipeline.cache import DiskCache
    
        obj = DiskCache(... )
        print(obj)
    """
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)

    def compute_key(self, feature_name, audio_hash, config_hash):
        """
        Compute key from VoxAtlas inputs.
        
        This function participates in pipeline orchestration, dependency resolution, caching, or execution. Use it to move data through the feature-centric VoxAtlas workflow without changing feature semantics.
        
        Parameters
        ----------
        feature_name : object
            Fully qualified VoxAtlas feature name, such as ``acoustic.pitch.f0``.
        audio_hash : object
            Stable hash representing the waveform payload and sampling rate for cache addressing.
        config_hash : object
            Stable hash representing the resolved pipeline configuration for cache addressing.
        
        Returns
        -------
        object
            Computed value or structure produced from the supplied inputs.
        
        Examples
        --------
            obj = DiskCache(... )
            value = obj.compute_key(feature_name=..., audio_hash=..., config_hash=...)
            print(value)
        """
        key_input = f"{feature_name}:{audio_hash}:{config_hash}"
        return hashlib.sha256(key_input.encode("utf-8")).hexdigest()

    def _cache_path(self, feature_name, key):
        return self.cache_dir / feature_name / f"{key}.pkl"

    def exists(self, feature_name, key):
        """
        Provide the ``exists`` public API.
        
        This function participates in pipeline orchestration, dependency resolution, caching, or execution. Use it to move data through the feature-centric VoxAtlas workflow without changing feature semantics.
        
        Parameters
        ----------
        feature_name : object
            Fully qualified VoxAtlas feature name, such as ``acoustic.pitch.f0``.
        key : object
            Cache key associated with a specific feature, audio payload, and configuration state.
        
        Returns
        -------
        object
            Return value produced by this API.
        
        Examples
        --------
            obj = DiskCache(... )
            value = obj.exists(feature_name=..., key=...)
            print(value)
        """
        return self._cache_path(feature_name, key).exists()

    def load(self, feature_name, key):
        """
        Provide the ``load`` public API.
        
        This function participates in pipeline orchestration, dependency resolution, caching, or execution. Use it to move data through the feature-centric VoxAtlas workflow without changing feature semantics.
        
        Parameters
        ----------
        feature_name : object
            Fully qualified VoxAtlas feature name, such as ``acoustic.pitch.f0``.
        key : object
            Cache key associated with a specific feature, audio payload, and configuration state.
        
        Returns
        -------
        object
            Return value produced by this API.
        
        Examples
        --------
            obj = DiskCache(... )
            value = obj.load(feature_name=..., key=...)
            print(value)
        """
        with self._cache_path(feature_name, key).open("rb") as f:
            return pickle.load(f)

    def save(self, feature_name, key, result):
        """
        Provide the ``save`` public API.
        
        This function participates in pipeline orchestration, dependency resolution, caching, or execution. Use it to move data through the feature-centric VoxAtlas workflow without changing feature semantics.
        
        Parameters
        ----------
        feature_name : object
            Fully qualified VoxAtlas feature name, such as ``acoustic.pitch.f0``.
        key : object
            Cache key associated with a specific feature, audio payload, and configuration state.
        result : object
            Computed feature output object to store, cache, or return to callers.
        
        Returns
        -------
        object
            Return value produced by this API.
        
        Examples
        --------
            obj = DiskCache(... )
            value = obj.save(feature_name=..., key=..., result=...)
            print(value)
        """
        cache_path = self._cache_path(feature_name, key)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        with cache_path.open("wb") as f:
            pickle.dump(result, f)

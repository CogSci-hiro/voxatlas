from concurrent.futures import ProcessPoolExecutor, as_completed


def run_feature(feature_name, feature_input, params):
    """
    Execute one extractor for one feature name.

    Parameters
    ----------
    feature_name : str
        Registered feature name to execute.
    feature_input : FeatureInput
        Prepared stream input passed to the extractor.
    params : dict
        Resolved feature configuration.

    Returns
    -------
    object
        Feature output returned by the extractor.

    Notes
    -----
    This helper is used by both the sequential path and worker processes.

    Examples
    --------
    Usage example::

        output = run_feature("acoustic.pitch.f0", feature_input, params)
        print(output)
    """
    from voxatlas.registry.feature_registry import registry

    extractor_cls = registry.get(feature_name)
    extractor = extractor_cls()
    return extractor.compute(feature_input, params)


def parallel_execute_layer(layer, registry, feature_input, n_jobs, feature_params=None):
    """
    Execute a dependency layer sequentially or in parallel.

    Parameters
    ----------
    layer : list of str
        Feature names belonging to the same dependency layer.
    registry : FeatureRegistry
        Registry used to resolve extractor classes.
    feature_input : FeatureInput
        Shared input bundle for the current stream.
    n_jobs : int
        Maximum number of worker processes to use.
    feature_params : dict | None
        Optional mapping from feature name to resolved configuration.

    Returns
    -------
    dict
        Mapping from feature name to computed output.

    Notes
    -----
    Features in the same layer are assumed to have no unresolved
    interdependencies.

    Examples
    --------
    Usage example::

        outputs = parallel_execute_layer(layer, registry, feature_input, n_jobs=1)
        print(outputs.keys())
    """
    feature_params = feature_params or {}

    if n_jobs <= 1 or len(layer) <= 1:
        results = {}

        for feature_name in layer:
            extractor_cls = registry.get(feature_name)
            extractor = extractor_cls()
            results[feature_name] = extractor.compute(
                feature_input,
                feature_params.get(feature_name, {}),
            )

        return results

    results = {}

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {
            executor.submit(
                run_feature,
                feature_name,
                feature_input,
                feature_params.get(feature_name, {}),
            ): feature_name
            for feature_name in layer
        }

        for future in as_completed(futures):
            feature_name = futures[future]
            results[feature_name] = future.result()

    return results

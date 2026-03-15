class FeatureStore:
    """
    Store intermediate and final feature outputs for one pipeline run.

    The feature store is the shared lookup table used during dependency
    resolution. Extractors read dependency outputs from this object instead of
    recomputing upstream features.

    Examples
    --------
    Usage example::

        store = FeatureStore()
        store.add("acoustic.pitch.f0", output)
        print(store.exists("acoustic.pitch.f0"))
    """

    def __init__(self):
        self._results: dict[str, object] = {}

    def add(self, feature_name, result):
        """
        Add a computed output to the store.

        Parameters
        ----------
        feature_name : str
            Fully qualified feature name.
        result : object
            Output object returned by an extractor.

        Returns
        -------
        None
            The store is updated in place.

        Notes
        -----
        Adding the same feature name again overwrites the previous value.

        Examples
        --------
        Usage example::

            store.add("syntax.dependencies", output)
        """
        self._results[feature_name] = result

    def get(self, feature_name):
        """
        Retrieve a stored feature output.

        Parameters
        ----------
        feature_name : str
            Fully qualified feature name.

        Returns
        -------
        object
            Stored feature output.

        Raises
        ------
        KeyError
            Raised when the feature is not present.

        Examples
        --------
        Usage example::

            output = store.get("syntax.dependencies")
            print(output)
        """
        return self._results[feature_name]

    def exists(self, feature_name):
        """
        Check whether a feature has already been stored.

        Parameters
        ----------
        feature_name : str
            Fully qualified feature name.

        Returns
        -------
        bool
            ``True`` when the feature exists in the store.

        Examples
        --------
        Usage example::

            print(store.exists("lexical.frequency.lookup"))
        """
        return feature_name in self._results

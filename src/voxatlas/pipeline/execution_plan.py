class ExecutionPlan:
    """
    Represent a dependency-sorted feature execution plan.

    Parameters
    ----------
    layers : iterable of iterable of str
        Sequence of dependency layers. Features in the same layer can be
        executed independently.

    Returns
    -------
    ExecutionPlan
        Normalized execution plan.

    Notes
    -----
    The ``features`` attribute flattens the layer structure in execution order.

    Examples
    --------
    >>> from voxatlas.pipeline.execution_plan import ExecutionPlan
    >>> plan = ExecutionPlan([["a"], ["b", "c"]])
    >>> plan.features
    ['a', 'b', 'c']
    """

    def __init__(self, layers):
        self.layers = [list(layer) for layer in layers]
        self.features = [
            feature_name
            for layer in self.layers
            for feature_name in layer
        ]

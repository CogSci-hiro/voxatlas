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
    Usage example::

        plan = ExecutionPlan([["syntax.dependencies"], ["syntax.complexity.clause_depth"]])
        print(plan.features)
    """

    def __init__(self, layers):
        self.layers = [list(layer) for layer in layers]
        self.features = [
            feature_name
            for layer in self.layers
            for feature_name in layer
        ]

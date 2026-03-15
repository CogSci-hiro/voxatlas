import pandas as pd


class Units:
    """
    Container for hierarchical speech units (tables) for a single stream.

    VoxAtlas feature extractors operate on *unit tables* (frames, tokens,
    phonemes, syllables, words, etc.) that are time-aligned and optionally
    linked through parent-child identifiers. ``Units`` is a lightweight,
    backend-agnostic wrapper around those tables: it stores them, normalizes
    unit type names (singular/plural aliases), and provides a small set of
    convenience accessors (lookup, durations, parent/children grouping).

    This class intentionally does **not** enforce a rigid schema beyond what
    its helper methods require; extractors may expect additional columns such
    as ``label`` or ``token`` depending on the feature.

    Parameters
    ----------
    frames : pandas.DataFrame | None
        Frame-level table.
    tokens : pandas.DataFrame | None
        Token-level table.
    phonemes : pandas.DataFrame | None
        Phoneme-level table.
    syllables : pandas.DataFrame | None
        Syllable-level table.
    sentences : pandas.DataFrame | None
        Sentence-level table.
    words : pandas.DataFrame | None
        Word-level table.
    ipus : pandas.DataFrame | None
        Inter-pausal-unit table.
    turns : pandas.DataFrame | None
        Turn-level table.
    speaker : str | None
        Optional speaker label for the stream.

    Returns
    -------
    Units
        Hierarchical unit container for one stream.

    Attributes
    ----------
    frames, tokens, phonemes, syllables, sentences, words, ipus, turns : pandas.DataFrame | None
        Stored unit tables. Any table may be ``None`` if it is unavailable.
    speaker : str | None
        Speaker label associated with this stream (if known).

    Notes
    -----
    **Unit labels**
    Methods that accept a ``unit_type`` (for example, :meth:`table`) accept
    both singular and plural labels:

    - ``"frame"`` / ``"frames"``
    - ``"token"`` / ``"tokens"``
    - ``"phoneme"`` / ``"phonemes"``
    - ``"syllable"`` / ``"syllables"``
    - ``"sentence"`` / ``"sentences"``
    - ``"word"`` / ``"words"``
    - ``"ipu"`` / ``"ipus"``
    - ``"turn"`` / ``"turns"``

    **Table conventions**
    ``Units`` works best when each DataFrame follows a few simple conventions:

    - ``id``: unique identifier for the unit row (typically integer-like).
    - ``start`` and ``end``: segment boundaries on a shared timeline
      (commonly seconds). Used by :meth:`duration` and by many extractors.
    - Parent-child links (optional): to connect units explicitly, include an
      ``<parent>_id`` column on the child table. For example, syllables that
      belong to words can carry a ``word_id`` column; phonemes that belong to
      syllables can carry a ``syllable_id`` column. :meth:`parent` and
      :meth:`children` use this naming convention.

    - :meth:`table` returns the underlying DataFrame object. If you mutate it,
      you are mutating the table stored on the ``Units`` instance.
    - If a requested table is missing (``None``), :meth:`table` raises
      ``ValueError``; callers can either catch this or check the relevant
      attribute first.

    Examples
    --------
    >>> import pandas as pd
    >>> from voxatlas.units import Units
    >>> words = pd.DataFrame({"id": [1], "start": [0.0], "end": [1.0], "label": ["hello"]})
    >>> syllables = pd.DataFrame(
    ...     {"id": [10], "word_id": [1], "start": [0.0], "end": [0.5], "label": ["he"]}
    ... )
    >>> units = Units(words=words, syllables=syllables, speaker="A")
    >>> units.table("word").shape
    (1, 4)
    >>> float(units.duration("word").iloc[0])
    1.0
    """

    _UNIT_ATTRS = {
        "frame": "frames",
        "frames": "frames",
        "token": "tokens",
        "tokens": "tokens",
        "phoneme": "phonemes",
        "phonemes": "phonemes",
        "syllable": "syllables",
        "syllables": "syllables",
        "sentence": "sentences",
        "sentences": "sentences",
        "word": "words",
        "words": "words",
        "ipu": "ipus",
        "ipus": "ipus",
        "turn": "turns",
        "turns": "turns",
    }

    def __init__(
        self,
        frames: pd.DataFrame | None = None,
        tokens: pd.DataFrame | None = None,
        phonemes: pd.DataFrame | None = None,
        syllables: pd.DataFrame | None = None,
        sentences: pd.DataFrame | None = None,
        words: pd.DataFrame | None = None,
        ipus: pd.DataFrame | None = None,
        turns: pd.DataFrame | None = None,
        speaker: str | None = None,
    ):
        self.frames = frames
        self.tokens = tokens
        self.phonemes = phonemes
        self.syllables = syllables
        self.sentences = sentences
        self.words = words
        self.ipus = ipus
        self.turns = turns
        self.speaker = speaker

    @classmethod
    def _normalize_unit_type(cls, unit_type: str) -> str:
        try:
            return cls._UNIT_ATTRS[unit_type]
        except KeyError as exc:
            raise ValueError(f"Invalid unit type: {unit_type}") from exc

    def table(self, unit_type: str) -> pd.DataFrame:
        """
        Return the table for a requested unit type.

        Parameters
        ----------
        unit_type : str
            Unit label such as ``"token"`` or ``"syllable"``.

        Returns
        -------
        pandas.DataFrame
            Table associated with the requested unit type.

        Raises
        ------
        ValueError
            Raised when the unit type is invalid or unavailable.

        Examples
        --------
        >>> import pandas as pd
        >>> from voxatlas.units import Units
        >>> tokens = pd.DataFrame({"id": [1], "start": [0.0], "end": [0.2], "label": ["hi"]})
        >>> units = Units(tokens=tokens)
        >>> units.table("token").columns.tolist()
        ['id', 'start', 'end', 'label']
        """
        table_name = self._normalize_unit_type(unit_type)
        table = getattr(self, table_name)

        if table is None:
            raise ValueError(f"No table available for unit type: {unit_type}")

        return table

    def get(self, unit_type: str) -> pd.DataFrame:
        """
        Alias for :meth:`table`.

        Parameters
        ----------
        unit_type : str
            Requested unit label.

        Returns
        -------
        pandas.DataFrame
            Requested unit table.

        Examples
        --------
        >>> import pandas as pd
        >>> from voxatlas.units import Units
        >>> words = pd.DataFrame({"id": [1], "start": [0.0], "end": [1.0], "label": ["hello"]})
        >>> units = Units(words=words)
        >>> units.get("word").shape[0]
        1
        """
        return self.table(unit_type)

    def duration(self, unit_type: str) -> pd.Series:
        """
        Compute durations from ``start`` and ``end`` columns.

        Parameters
        ----------
        unit_type : str
            Requested unit label.

        Returns
        -------
        pandas.Series
            Duration values for each row.

        Examples
        --------
        >>> import pandas as pd
        >>> from voxatlas.units import Units
        >>> words = pd.DataFrame({"id": [1], "start": [0.25], "end": [1.00], "label": ["hello"]})
        >>> units = Units(words=words)
        >>> float(units.duration("word").iloc[0])
        0.75
        """
        table = self.table(unit_type)
        return table["end"] - table["start"]

    def parent(self, child_type: str, parent_type: str) -> pd.Series:
        """
        Return parent identifiers for a child unit table.

        Parameters
        ----------
        child_type : str
            Child unit label.
        parent_type : str
            Parent unit label.

        Returns
        -------
        pandas.Series
            Parent identifier column.

        Raises
        ------
        ValueError
            Raised when the mapping column is unavailable.

        Examples
        --------
        >>> import pandas as pd
        >>> from voxatlas.units import Units
        >>> words = pd.DataFrame({"id": [1], "start": [0.0], "end": [1.0], "label": ["hello"]})
        >>> syllables = pd.DataFrame(
        ...     {"id": [10, 11], "word_id": [1, 1], "start": [0.0, 0.5], "end": [0.5, 1.0], "label": ["he", "llo"]}
        ... )
        >>> units = Units(words=words, syllables=syllables)
        >>> units.parent("syllable", "word").tolist()
        [1, 1]
        """
        child_table = self.table(child_type)
        parent_column = f"{self._normalize_unit_type(parent_type)[:-1]}_id"

        if parent_column not in child_table.columns:
            raise ValueError(
                f"No parent mapping from {child_type} to {parent_type}"
            )

        return child_table[parent_column]

    def children(self, parent_type: str, child_type: str):
        """
        Group child units by parent identifier.

        Parameters
        ----------
        parent_type : str
            Parent unit label.
        child_type : str
            Child unit label.

        Returns
        -------
        DataFrameGroupBy
            Grouped child table keyed by parent identifier.

        Raises
        ------
        ValueError
            Raised when the mapping column is unavailable.

        Examples
        --------
        >>> import pandas as pd
        >>> from voxatlas.units import Units
        >>> words = pd.DataFrame({"id": [1], "start": [0.0], "end": [1.0], "label": ["hello"]})
        >>> phonemes = pd.DataFrame(
        ...     {"id": [100, 101], "word_id": [1, 1], "start": [0.0, 0.5], "end": [0.5, 1.0], "label": ["h", "i"]}
        ... )
        >>> units = Units(words=words, phonemes=phonemes)
        >>> units.children("word", "phoneme").ngroups
        1
        """
        child_table = self.table(child_type)
        parent_column = f"{self._normalize_unit_type(parent_type)[:-1]}_id"

        if parent_column not in child_table.columns:
            raise ValueError(
                f"No child mapping from {parent_type} to {child_type}"
            )

        return child_table.groupby(parent_column)

    def group(self, child_type: str, by: str):
        """
        Alias for :meth:`children` using ``by`` as the parent unit.

        Parameters
        ----------
        child_type : str
            Child unit label.
        by : str
            Parent unit label.

        Returns
        -------
        DataFrameGroupBy
            Grouped child table.

        Examples
        --------
        >>> import pandas as pd
        >>> from voxatlas.units import Units
        >>> words = pd.DataFrame({"id": [1], "start": [0.0], "end": [1.0], "label": ["hello"]})
        >>> phonemes = pd.DataFrame(
        ...     {"id": [100, 101], "word_id": [1, 1], "start": [0.0, 0.5], "end": [0.5, 1.0], "label": ["h", "i"]}
        ... )
        >>> units = Units(words=words, phonemes=phonemes)
        >>> units.group("phoneme", by="word").ngroups
        1
        """
        return self.children(by, child_type)

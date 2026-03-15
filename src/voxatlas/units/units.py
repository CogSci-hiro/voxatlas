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

    Unit Labels
    -----------
    Methods that accept a ``unit_type`` (for example :meth:`table`) understand
    both singular and plural labels:

    - ``"frame"`` / ``"frames"``
    - ``"token"`` / ``"tokens"``
    - ``"phoneme"`` / ``"phonemes"``
    - ``"syllable"`` / ``"syllables"``
    - ``"sentence"`` / ``"sentences"``
    - ``"word"`` / ``"words"``
    - ``"ipu"`` / ``"ipus"``
    - ``"turn"`` / ``"turns"``

    Table Conventions
    -----------------
    ``Units`` works best when each DataFrame follows a few simple conventions:

    - ``id``: unique identifier for the unit row (typically integer-like).
    - ``start`` and ``end``: segment boundaries on a shared timeline
      (commonly seconds). Used by :meth:`duration` and by many extractors.
    - Parent-child links (optional): to connect units explicitly, include an
      ``<parent>_id`` column on the child table. For example, syllables that
      belong to words can carry a ``word_id`` column; phonemes that belong to
      syllables can carry a ``syllable_id`` column. :meth:`parent` and
      :meth:`children` use this naming convention.

    Notes
    -----
    - :meth:`table` returns the underlying DataFrame object. If you mutate it,
      you are mutating the table stored on the ``Units`` instance.
    - If a requested table is missing (``None``), :meth:`table` raises
      ``ValueError``; callers can either catch this or check the relevant
      attribute first.

    Examples
    --------
    Minimal usage::

        units = Units(words=word_table, syllables=syllable_table, speaker="A")
        print(units.table("word"))

    Computing durations::

        word_durations = units.duration("word")  # end - start

    Grouping children by explicit parent ids::

        # Requires a ``word_id`` column on the phoneme table.
        phonemes_by_word = units.children("word", "phoneme")
        first_word_phonemes = phonemes_by_word.get_group(10)
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
        Usage example::

            token_table = units.table("token")
            print(token_table.head())
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
        Usage example::

            words = units.get("word")
            print(words.columns)
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
        Usage example::

            durations = units.duration("word")
            print(durations.head())
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
        Usage example::

            word_ids = units.parent("syllable", "word")
            print(word_ids.head())
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
        Usage example::

            grouped = units.children("word", "phoneme")
            print(grouped.ngroups)
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
        Usage example::

            grouped = units.group("phoneme", by="word")
            print(grouped.ngroups)
        """
        return self.children(by, child_type)

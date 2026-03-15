from pathlib import Path

import pandas as pd


def _parse_quoted_value(line: str) -> str:
    _, value = line.split("=", 1)
    value = value.strip()

    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]

    return value


def load_textgrid(path: str | Path) -> dict[str, pd.DataFrame]:
    """
    Parse a Praat TextGrid file into per-tier interval tables.

    Each returned DataFrame contains interval rows with ``id``, ``start``,
    ``end``, and ``label`` columns. Tier names are used as dictionary keys.

    Parameters
    ----------
    path : str | Path
        Path to a TextGrid file on disk.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Mapping from tier name to interval table.

    Notes
    -----
    This parser targets interval tiers (``intervals [n]`` blocks). Point tiers
    are not expanded into the output structure.

    Examples
    --------
    >>> tiers = load_textgrid("alignment.TextGrid")
    >>> sorted(tiers.keys())  # doctest: +SKIP
    ['phones', 'words']
    >>> tiers["words"].columns.tolist()  # doctest: +SKIP
    ['id', 'start', 'end', 'label']
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    tiers: dict[str, pd.DataFrame] = {}
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if not line.startswith("item ["):
            i += 1
            continue

        tier_name = None
        intervals = []
        i += 1

        while i < len(lines):
            line = lines[i].strip()

            if line.startswith("item ["):
                break

            if line.startswith("name ="):
                tier_name = _parse_quoted_value(line)
                i += 1
                continue

            if line.startswith("intervals ["):
                interval = {
                    "id": len(intervals) + 1,
                    "start": None,
                    "end": None,
                    "label": "",
                }
                i += 1

                while i < len(lines):
                    line = lines[i].strip()

                    if line.startswith("intervals [") or line.startswith("item ["):
                        break

                    if line.startswith("xmin ="):
                        interval["start"] = float(line.split("=", 1)[1].strip())
                    elif line.startswith("xmax ="):
                        interval["end"] = float(line.split("=", 1)[1].strip())
                    elif line.startswith("text ="):
                        interval["label"] = _parse_quoted_value(line)

                    i += 1

                intervals.append(interval)
                continue

            i += 1

        if tier_name is not None:
            tiers[tier_name] = pd.DataFrame(intervals)

    return tiers

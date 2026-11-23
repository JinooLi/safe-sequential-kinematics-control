"""Versioned output-index mapping for Panda link selection.

The first ten indices retain their historical meaning. A Direct action is an
explicit choice with no intermediate frames, not a missing prediction.
"""

from itertools import combinations

import numpy as np

LINK_PAIRS = tuple(combinations(range(2, 7), 2))
LINK_ACTIONS = LINK_PAIRS + tuple((link,) for link in range(2, 7)) + ((),)


def validate_actions(actions):
    actions = tuple(tuple(links) for links in actions)
    if any(
        isinstance(link, (bool, np.bool_)) or not isinstance(link, (int, np.integer))
        for links in actions
        for link in links
    ) or actions not in (LINK_PAIRS, LINK_ACTIONS):
        raise ValueError("unexpected output-index/action mapping")
    return actions


def action_id(links):
    links = tuple(links)
    if links not in LINK_ACTIONS:
        raise ValueError("unsupported link selection")
    return "_".join(f"link{link}" for link in links) if links else "direct"


def catalog_metadata(actions=LINK_ACTIONS):
    actions = validate_actions(actions)
    return {
        "action_schema_version": 1 if actions == LINK_PAIRS else 2,
        "actions": [
            {
                "action_id": action_id(links),
                "method": "sequential" if links else "direct",
                "stage_frames": [f"link{link}" for link in links],
            }
            for links in actions
        ],
    }


def catalog_arrays(actions=LINK_ACTIONS):
    actions = validate_actions(actions)
    return {
        "action_schema_version": np.asarray(1 if actions == LINK_PAIRS else 2),
        "action_ids": np.asarray([action_id(links) for links in actions]),
        "stage_links": np.asarray(
            [list(links) + [0] * (2 - len(links)) for links in actions], dtype=int
        ),
        "stage_count": np.asarray([len(links) for links in actions], dtype=int),
    }


def actions_from_arrays(data):
    """Read safe, non-object NPZ metadata; never infer a mapping from width alone."""
    keys = ("action_schema_version", "action_ids", "stage_links", "stage_count")
    if not any(key in data for key in keys):
        if "link_pairs" not in data or not np.array_equal(
            data["link_pairs"], LINK_PAIRS
        ):
            raise ValueError("missing or unexpected output-index/link-pair mapping")
        return LINK_PAIRS
    if not all(key in data for key in keys):
        raise ValueError("incomplete action catalog")
    version = np.asarray(data["action_schema_version"])
    if (
        version.shape != ()
        or version.dtype.kind not in "iu"
        or version.item() not in (1, 2)
    ):
        raise ValueError("unsupported action schema version")
    actions = LINK_PAIRS if version.item() == 1 else LINK_ACTIONS
    expected = catalog_arrays(actions)
    if any(not np.array_equal(data[key], expected[key]) for key in keys):
        raise ValueError("unexpected output-index/action mapping")
    if "link_pairs" in data and (
        actions != LINK_PAIRS or not np.array_equal(data["link_pairs"], LINK_PAIRS)
    ):
        raise ValueError("conflicting legacy and action catalogs")
    return actions


def actions_from_metadata(record):
    """Validate JSON/checkpoint catalogs, including historical ten-pair records."""
    if "actions" not in record and "action_schema_version" not in record:
        if record.get("link_pairs") != [list(pair) for pair in LINK_PAIRS]:
            raise ValueError("missing or unexpected output-index/link-pair mapping")
        return LINK_PAIRS
    version = record.get("action_schema_version")
    if type(version) is not int or version not in (1, 2):
        raise ValueError("unsupported action schema version")
    actions = LINK_PAIRS if version == 1 else LINK_ACTIONS
    if record.get("actions") != catalog_metadata(actions)["actions"]:
        raise ValueError("unexpected output-index/action mapping")
    if "link_pairs" in record and (
        actions != LINK_PAIRS
        or record["link_pairs"] != [list(pair) for pair in LINK_PAIRS]
    ):
        raise ValueError("conflicting legacy and action catalogs")
    return actions

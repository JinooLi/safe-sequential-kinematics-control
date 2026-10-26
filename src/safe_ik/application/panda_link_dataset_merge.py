"""Verify and merge disjoint GPU-group campaigns without running simulations."""

import argparse
import json
from pathlib import Path

from .panda_comparison import atomic_json, digest
from .panda_experiment_configuration import load_panda_experiment
from .panda_link_dataset import export_dataset, utc_now
from .panda_link_dataset_campaign import campaign_plan, identity, verified_batch


def merge_groups(directories, output):
    contexts, rows, groups, requests, identities, inputs = [], [], [], [], [], {}
    for directory in map(Path, directories):
        complete = json.loads((directory / "complete.json").read_text())
        for name, expected in complete["data_sha256"].items():
            if digest(directory / name) != expected:
                raise ValueError(f"group data hash mismatch: {directory / name}")
        campaign = json.loads((directory / "campaign.json").read_text())
        if campaign["status"] != "completed":
            raise ValueError("all GPU groups must be completed")
        request = dict(campaign["request"])
        groups.append(request.pop("group_index", 0))
        requests.append(request)
        identities.append(campaign["identity"])
        expected_plan = campaign_plan(campaign["request"])
        if [(b["index"], b["seed"], b["contexts"]) for b in campaign["batches"]] != [
            (b["index"], b["seed"], b["contexts"]) for b in expected_plan
        ]:
            raise ValueError("group batch plan mismatch")
        for batch in campaign["batches"]:
            attempt = directory / batch["attempts"][-1]["output"]
            if (
                batch["status"] != "completed"
                or digest(attempt / "complete.json") != batch["complete_sha256"]
            ):
                raise ValueError("group batch completion mismatch")
            ctx, res = verified_batch(
                attempt,
                batch["contexts"],
                identity(load_panda_experiment(directory / batch["config"])),
            )
            contexts.extend(ctx)
            rows.extend(res)
        inputs[str(directory.resolve())] = digest(directory / "complete.json")
    if (
        not groups
        or any(r != requests[0] for r in requests)
        or any(i != identities[0] for i in identities)
    ):
        raise ValueError("GPU group configuration/source mismatch")
    if sorted(groups) != list(range(requests[0].get("groups", 1))):
        raise ValueError("missing or duplicate GPU groups")
    if len({c["case_id"] for c in contexts}) != len(contexts):
        raise ValueError("duplicate scene identity across GPU groups")
    if sum(c["ik_available"] for c in contexts) != requests[0]["total_contexts"]:
        raise ValueError("global context count mismatch")
    # Only create output after validation; never overwrite old exports.
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    counts = export_dataset(output, contexts, rows)
    atomic_json(
        output / "complete.json",
        dict(
            **counts,
            finished_utc=utc_now(),
            groups=inputs,
            data_sha256={
                name: digest(output / name)
                for name in (
                    "dataset.npz",
                    "contexts.jsonl",
                    "results.json",
                    "results.csv",
                )
            },
        ),
    )
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge_groups(args.groups, args.output)))


if __name__ == "__main__":
    main()

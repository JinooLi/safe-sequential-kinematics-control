"""Parent-process tqdm display; numerical code emits UI-independent events."""

from time import monotonic

from tqdm import tqdm

from ..domain.ik_contracts import IKProgress


class PandaProgress:
    def __init__(self, *, enabled=True, file=None):
        self.enabled = enabled
        self.file = file
        self.bar = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.finish(
            "interrupted"
            if exc_type is KeyboardInterrupt
            else "error" if exc_type else "finished"
        )

    def start(self, description, total, unit):
        self.finish("finished")
        self.last_refresh = 0.0
        self.bar = tqdm(
            total=total,
            desc=description,
            unit=unit,
            dynamic_ncols=True,
            mininterval=0.1,
            disable=not self.enabled,
            file=self.file,
        )

    def update(self, completed, detail=""):
        if self.bar is not None:
            self.bar.set_postfix_str(detail, refresh=False)
            self.bar.update(max(0, completed - self.bar.n))
            # Completed seeds may not change for a long time; still refresh
            # iteration telemetry at a bounded rate while workers are active.
            now = monotonic()
            if now - self.last_refresh >= 0.1:
                self.bar.refresh()
                self.last_refresh = now

    def start_ik(self, seed_count, iteration_limit, require_safe):
        self.iterations = {}
        self.finished = set()
        self.candidates = set()
        self.iteration_budget = seed_count * iteration_limit
        self.start(
            "Safe IK" if require_safe else "Pose-only IK diagnostic", seed_count, "seed"
        )
        self.bar.set_postfix_str("starting workers / waiting for first iteration")

    def ik_update(self, event: IKProgress):
        self.iterations[event.seed_index] = max(
            self.iterations.get(event.seed_index, 0), event.iteration
        )
        if event.finished:
            self.finished.add(event.seed_index)
        if event.candidate_found:
            self.candidates.add(event.seed_index)
        self.update(
            len(self.finished),
            f"iter={sum(self.iterations.values())}/{self.iteration_budget} max, "
            f"seed={event.seed_index + 1}:{self.iterations[event.seed_index]}, "
            f"found={len(self.candidates)}",
        )

    def finish(self, status):
        if self.bar is not None:
            # Early convergence/failure is not fabricated as 100% of the budget.
            self.bar.set_description(f"{self.bar.desc} [{status}]", refresh=False)
            self.bar.close()
            self.bar = None

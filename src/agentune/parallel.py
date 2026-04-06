"""Parallel trial execution using concurrent.futures."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import optuna

logger = logging.getLogger(__name__)


class ParallelOptimizer:
    """Run Optuna trials in parallel using a thread pool.

    Optuna's RDBStorage is thread-safe — multiple threads can call
    study.optimize(n_trials=1) concurrently on the same study.
    """

    def __init__(self, n_jobs: int = 1) -> None:
        self._n_jobs = max(1, n_jobs)

    def optimize(
        self,
        study: optuna.Study,
        objective: Callable[[optuna.Trial], float],
        n_trials: int,
        timeout: float | None = None,
    ) -> None:
        if self._n_jobs <= 1:
            study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
            return

        start = time.monotonic()
        remaining = n_trials

        def _worker(worker_trials: int) -> int:
            worker_timeout = None
            if timeout is not None:
                elapsed = time.monotonic() - start
                worker_timeout = max(0.1, timeout - elapsed)
            study.optimize(
                objective,
                n_trials=worker_trials,
                timeout=worker_timeout,
                show_progress_bar=False,
            )
            return worker_trials

        with ThreadPoolExecutor(max_workers=self._n_jobs) as executor:
            while remaining > 0:
                if timeout is not None and (time.monotonic() - start) >= timeout:
                    break

                batch = min(remaining, self._n_jobs)
                trials_per_worker = max(1, batch // self._n_jobs)
                futures = []
                for _ in range(min(batch, self._n_jobs)):
                    chunk = min(trials_per_worker, remaining)
                    if chunk <= 0:
                        break
                    futures.append(executor.submit(_worker, chunk))
                    remaining -= chunk

                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception:
                        logger.warning("Trial worker failed", exc_info=True)

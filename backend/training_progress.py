"""Estimate only an active training loop from observed advancing iterations."""
from __future__ import annotations
from collections import deque
import math
import re
import statistics
import time

ETA_MAX_AGE_SECONDS = 45


class TrainingProgress:
    def __init__(self):
        self.reset()

    def reset(self):
        self.last_iteration = None
        self.total = None
        self.first_at = None
        self.last_at = None
        self.rates = deque(maxlen=5)
        self.intervals = deque(maxlen=5)
        self.latest = None

    def update(self, message, now=None):
        # Do not interpret camera-loading counters, log filenames or plan budgets.
        match = re.search(r'(?:Training progress:|Optimizing splats)\s*.*?(\d+)\s*/\s*(\d+)', str(message))
        if not match:
            self.reset()
            return None
        current, total = map(int, match.groups())
        if not 0 <= current < total:
            self.reset()
            return None
        clock = time.monotonic() if now is None else float(now)
        timestamp = time.time() if now is None else float(now)
        if total != self.total or self.last_iteration is not None and current < self.last_iteration:
            self.reset()
        if self.last_iteration is not None and current == self.last_iteration:
            # tqdm postfix changes can repeat the same counter; this is not progress.
            return self.latest
        if self.last_at is not None and clock - self.last_at > ETA_MAX_AGE_SECONDS:
            self.reset()
        if self.first_at is None:
            self.first_at = clock
        if self.last_at is not None and clock > self.last_at:
            self.intervals.append((current - self.last_iteration) / (clock - self.last_at))
        rate_match = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*(it/s|s/it)', str(message))
        if rate_match:
            value = float(rate_match.group(1))
            if value > 0:
                self.rates.append(value if rate_match.group(2) == 'it/s' else 1 / value)
        else:
            self.rates.clear()
        self.last_iteration, self.total, self.last_at = current, total, clock

        # At least three advancing observations and two seconds, with no >50% spread.
        # Densification or pauses can change speed: suppress ETA until it stabilizes again.
        candidates = self.rates if len(self.rates) >= 3 else self.intervals
        stable = (clock - self.first_at >= 2 and len(candidates) >= 3
                  and min(candidates) > 0 and max(candidates) / min(candidates) <= 1.5)
        rate = statistics.median(candidates) if stable else None
        eta = (total - current) / rate if rate else None
        if eta is not None and not math.isfinite(eta):
            eta = rate = None
        self.latest = {'phase': 'training', 'iteration': current, 'total': total,
            'updated_at': timestamp, 'eta_seconds': eta, 'iterations_per_second': rate,
            'eta_basis': ('tqdm_observed_rate' if candidates is self.rates else 'observed_iteration_intervals') if stable else 'collecting',
            'scope': 'training_loop_only'}
        return self.latest


def fresh_training_progress(progress, status, now=None):
    if status != 'running' or not progress:
        return None
    result = dict(progress)
    age = (time.time() if now is None else float(now)) - result.get('updated_at', 0)
    if age < 0 or age > ETA_MAX_AGE_SECONDS:
        result.update(phase='stale', eta_seconds=None, iterations_per_second=None)
    return result

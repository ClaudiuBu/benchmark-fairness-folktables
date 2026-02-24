"""Progress tracking utilities for benchmark runs."""

import sys
import time
from typing import Optional


class ProgressTracker:
    """Tracks and displays progress of long-running benchmarks.
    
    Supports both tqdm (if available) and fallback text output.
    """

    def __init__(self, total: int, enabled: bool = True):
        """Initialize progress tracker.
        
        Args:
            total: Total number of steps expected
            enabled: Whether to display progress (can be disabled for quiet mode)
        """
        self.total = max(int(total), 1)
        self.count = 0
        self.enabled = enabled
        self._tqdm = None
        self._start_time = time.time()

        if not self.enabled:
            return

        try:
            from tqdm import tqdm  # type: ignore
            self._tqdm = tqdm(total=self.total, desc="Benchmark", unit="step")
        except Exception:
            # tqdm not available, will use text output
            self._tqdm = None

    def update(self, label: str = ""):
        """Update progress by one step.
        
        Args:
            label: Optional descriptive label for current step
        """
        if not self.enabled:
            return

        self.count += 1
        if self._tqdm is not None:
            if label:
                self._tqdm.set_postfix_str(label)
            self._tqdm.update(1)
        else:
            self._print_progress(label)

    def _print_progress(self, label: str = ""):
        """Print text-based progress output with ETA."""
        pct = int(round(100 * self.count / self.total))
        elapsed = time.time() - self._start_time
        avg = elapsed / max(self.count, 1)
        remaining = avg * (self.total - self.count)
        eta_sec = int(round(remaining))
        
        msg = f"Progress: {self.count}/{self.total} ({pct}%)"
        msg += f" - ETA ~{eta_sec}s"
        if label:
            msg += f" - {label}"
        print(msg, file=sys.stderr)

    def close(self):
        """Finalize progress tracking (close tqdm if used)."""
        if self._tqdm is not None:
            self._tqdm.close()

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.close()


class ProgressCalculator:
    """Calculates total progress steps needed for different benchmark modes."""

    @staticmethod
    def calculate_validation_steps(
        num_seeds: int,
        num_methods: int,
        num_periods: int,
        maintenance_strategies: list,
    ) -> int:
        """Calculate progression steps for temporal validation/testing.
        
        Args:
            num_seeds: Number of random seeds
            num_methods: Number of fairness methods
            num_periods: Number of time periods (e.g., years)
            maintenance_strategies: List of strategies ('no-retrain', 'retrain')
        
        Returns:
            Total number of progress steps
        """
        total = 0
        
        # no-retrain: train once, test on aggregated set + test on each period
        if "no-retrain" in maintenance_strategies:
            total += num_seeds * num_methods  # aggregated test
            total += num_seeds * num_methods * num_periods  # per-period tests
        
        # retrain: for each period, retrain then test
        if "retrain" in maintenance_strategies:
            total += num_seeds * num_methods * num_periods
        
        return total

    @staticmethod
    def calculate_static_steps(
        num_seeds: int,
        num_methods: int,
    ) -> int:
        """Calculate progress steps for static benchmarks.
        
        Args:
            num_seeds: Number of random seeds
            num_methods: Number of fairness methods
        
        Returns:
            Total number of progress steps
        """
        return num_seeds * num_methods

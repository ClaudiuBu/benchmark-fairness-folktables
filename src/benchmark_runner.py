"""Thin CLI wrapper for the Folktables benchmark runner."""

from src.benchmark.runner import run_benchmark


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.benchmark_runner <config.yaml> [<config.yaml> ...]")
        sys.exit(1)

    for config_path in sys.argv[1:]:
        run_benchmark(config_path)

"""Migrate legacy flat results directories to config-grouped timestamped structure.

Usage:
  python -m src.benchmark.migrate_legacy_results
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path("results/folktables")
TIMESTAMP_RE = re.compile(r"^\d{8}_\d{6}$")
CONFIG_TS_RE = re.compile(r"^config_(\d{8}_\d{6})\.ya?ml$")


def _sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "legacy_config"


def _looks_like_base_dir(path: Path) -> bool:
    if (path / "run_history.csv").exists():
        return True
    return any(child.is_dir() and TIMESTAMP_RE.match(child.name) for child in path.iterdir())


def _choose_base_dir(result_dir: Path) -> Path:
    candidates = [
        child
        for child in result_dir.iterdir()
        if child.is_dir() and not child.name.startswith(".") and not TIMESTAMP_RE.match(child.name)
    ]
    for candidate in candidates:
        if _looks_like_base_dir(candidate):
            return candidate

    run_meta_path = result_dir / "run_meta.json"
    if run_meta_path.exists():
        try:
            run_meta = json.loads(run_meta_path.read_text())
            experiment = str(run_meta.get("experiment", "")).strip()
            if experiment:
                return result_dir / _sanitize_name(experiment)
        except Exception:
            pass

    return result_dir / _sanitize_name(result_dir.name)


def _choose_legacy_run_id(root_files: list[Path]) -> str:
    timestamps = []
    for file_path in root_files:
        match = CONFIG_TS_RE.match(file_path.name)
        if match:
            timestamps.append(match.group(1))

    if timestamps:
        return f"legacy_{sorted(timestamps)[-1]}"
    return f"legacy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _append_history(base_dir: Path, run_dir: Path, result_dir: Path):
    run_meta = {}
    run_meta_path = run_dir / "run_meta.json"
    if run_meta_path.exists():
        try:
            run_meta = json.loads(run_meta_path.read_text())
        except Exception:
            run_meta = {}

    row = {
        "run_id": run_dir.name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config_path": run_meta.get("config_path", ""),
        "experiment": run_meta.get("experiment", result_dir.name),
        "task": run_meta.get("task", ""),
        "mode": "",
        "output_dir": str(run_dir),
        "results": str(run_dir / "benchmark_results.csv") if (run_dir / "benchmark_results.csv").exists() else "",
        "summary": str(run_dir / "benchmark_summary.csv") if (run_dir / "benchmark_summary.csv").exists() else "",
        "elapsed_seconds": run_meta.get("elapsed_seconds", ""),
    }

    history_path = base_dir / "run_history.csv"
    row_df = pd.DataFrame([row])
    if history_path.exists():
        row_df.to_csv(history_path, mode="a", header=False, index=False)
    else:
        row_df.to_csv(history_path, index=False)


def migrate() -> int:
    if not ROOT.exists():
        print(f"No results root found: {ROOT}")
        return 0

    migrated_count = 0

    for result_dir in sorted(path for path in ROOT.iterdir() if path.is_dir()):
        if not (result_dir / "benchmark_results.csv").exists():
            continue

        root_files = [
            file_path
            for file_path in result_dir.iterdir()
            if file_path.is_file() and file_path.name != "run_history.csv"
        ]

        if not root_files:
            continue

        base_dir = _choose_base_dir(result_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        run_id = _choose_legacy_run_id(root_files)
        run_dir = base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        moved_files = 0
        for source in root_files:
            destination = run_dir / source.name
            if destination.exists():
                stem, suffix = source.stem, source.suffix
                candidate = run_dir / f"{stem}_migrated{suffix}"
                index = 1
                while candidate.exists():
                    candidate = run_dir / f"{stem}_migrated_{index}{suffix}"
                    index += 1
                destination = candidate

            source.rename(destination)
            moved_files += 1

        _append_history(base_dir, run_dir, result_dir)

        migrated_count += 1
        print(f"Migrated {result_dir} -> {run_dir} ({moved_files} files)")

    print(f"Done. Migrated directories: {migrated_count}")
    return migrated_count


if __name__ == "__main__":
    migrate()

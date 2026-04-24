#!/usr/bin/env python3
"""
Full benchmark runner for both Income and Employment tasks.
Runs temporal fairness benchmarks with all methods and seeds.
Then generates fairness gap tables and syncs plots to paper.

Usage:
    python run_full_benchmark.py
"""

import sys
import time
import subprocess
import re
from pathlib import Path
from datetime import datetime

from src.benchmark.runner import run_benchmark


def update_latex_with_run_ids(latex_file: Path, run_ids: dict) -> bool:
    """Update LaTeX file with latest run IDs.
    
    Args:
        latex_file: Path to articol.tex
        run_ids: Dict with task names as keys, each containing {'run_id': '...', 'output_dir': '...'}
    
    Returns:
        True if updated successfully
    """
    try:
        content = latex_file.read_text()
        original_content = content
        
        # Process each task's run information
        for task_name, run_info in run_ids.items():
            output_dir = Path(run_info['output_dir'])
            run_id = run_info.get('run_id', '')
            
            # Calculate relative path from paper/ directory to output_dir
            try:
                # Get relative path from paper directory
                paper_dir = latex_file.parent
                rel_path = output_dir.relative_to(paper_dir.parent)
                rel_path_str = f"../{str(rel_path).replace(chr(92), '/')}"  # Path relative to paper/
                
                # Replace paths for summary files specific to this task
                pattern = r'\\input\{[^}]*summary_' + re.escape(task_name) + r'_[^}]*\.tex\}'
                
                # Find all matches to extract which attributes are present
                matches = list(re.finditer(pattern, content))
                for match in matches:
                    old_path = match.group(0)
                    # Extract the attribute from the old path
                    attr_match = re.search(r'summary_' + re.escape(task_name) + r'_(\w+)\.tex', old_path)
                    if attr_match:
                        attr = attr_match.group(1)
                        new_path = f'\\input{{{rel_path_str}/summary_{task_name}_{attr}.tex}}'
                        content = content.replace(old_path, new_path, 1)
                        print(f"  Updated: summary_{task_name}_{attr}.tex")
            
            except ValueError as e:
                print(f"⚠️  Could not compute relative path for {task_name}: {e}")
        
        # Only write if content changed
        if content != original_content:
            latex_file.write_text(content)
            print(f"✅ Updated LaTeX file with new run IDs")
            return True
        else:
            print(f"⚠️  No changes needed in LaTeX file")
            return True
    
    except Exception as e:
        print(f"❌ Error updating LaTeX file: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run full benchmarks for both income and employment tasks."""
    
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    # Full benchmark configurations
    CONFIGS = [
        {
            "name": "Income (Temporal)",
            "path": "configs/folktables_benchmark_temporal.yaml",
            "description": "Income prediction task with temporal evaluation"
        },
        {
            "name": "Employment (Temporal)",
            "path": "configs/folktables_benchmark_temporal_employment.yaml",
            "description": "Employment status prediction task with temporal evaluation"
        }
    ]
    
    print("=" * 80)
    print("FULL BENCHMARK RUNNER - Income & Employment Tasks")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results = []
    run_ids = {}  # Store run info for LaTeX update
    total_start = time.time()
    
    for idx, config in enumerate(CONFIGS, 1):
        print(f"\n{'='*80}")
        print(f"[{idx}/{len(CONFIGS)}] {config['name']}")
        print(f"{'='*80}")
        print(f"Description: {config['description']}")
        print(f"Config: {config['path']}")
        print()
        
        config_path = project_root / config["path"]
        if not config_path.exists():
            print(f"❌ Config not found: {config_path}")
            results.append({
                "task": config["name"],
                "status": "FAILED",
                "error": "Config file not found",
                "duration": 0
            })
            continue
        
        try:
            task_start = time.time()
            print(f"⏱️  Starting benchmark run...")
            print()
            
            # Run benchmark and capture metadata
            run_metadata = run_benchmark(str(config_path))
            
            task_duration = time.time() - task_start
            print()
            print(f"✅ {config['name']} completed successfully")
            print(f"⏱️  Duration: {task_duration/3600:.2f} hours ({task_duration/60:.1f} minutes)")
            
            # Store run metadata for LaTeX update
            if run_metadata:
                task_key = run_metadata.get('task', 'unknown')
                run_ids[task_key] = run_metadata
            
            results.append({
                "task": config["name"],
                "status": "SUCCESS",
                "error": None,
                "duration": task_duration
            })
            
        except Exception as e:
            task_duration = time.time() - task_start
            print()
            print(f"❌ {config['name']} failed with error:")
            print(f"   {str(e)}")
            print(f"⏱️  Duration before failure: {task_duration/60:.1f} minutes")
            
            results.append({
                "task": config["name"],
                "status": "FAILED",
                "error": str(e),
                "duration": task_duration
            })
    
    # Summary report
    print(f"\n{'='*80}")
    print("SUMMARY - Benchmark Runs")
    print(f"{'='*80}")
    total_duration = time.time() - total_start
    
    for result in results:
        status_icon = "✅" if result["status"] == "SUCCESS" else "❌"
        duration_str = f"{result['duration']/3600:.2f}h" if result['duration'] >= 3600 else f"{result['duration']/60:.1f}m"
        print(f"{status_icon} {result['task']:30s} [{duration_str:>8s}] {result['status']}")
        if result['error']:
            print(f"   Error: {result['error']}")
    
    # Check if benchmarks succeeded before proceeding
    failed = sum(1 for r in results if r["status"] == "FAILED")
    if failed > 0:
        print()
        print(f"⚠️  {failed}/{len(results)} benchmark(s) failed. Skipping post-processing.")
        print(f"Total duration: {total_duration/3600:.2f} hours ({total_duration/60:.1f} minutes)")
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return 1
    
    # Step 1b: Update LaTeX file with new run IDs
    print(f"\n{'='*80}")
    print("STEP 1B: Update LaTeX References with New Run IDs")
    print(f"{'='*80}")
    print()
    
    if run_ids:
        latex_file = project_root / "paper" / "articol.tex"
        if latex_file.exists():
            update_latex_with_run_ids(latex_file, run_ids)
        else:
            print(f"⚠️  LaTeX file not found: {latex_file}")
    else:
        print("⚠️  No run IDs captured, skipping LaTeX update")
    
    # Step 2: Generate fairness gap tables
    print(f"\n{'='*80}")
    print("STEP 2: Generate Fairness Gap Tables")
    print(f"{'='*80}")
    print()
    
    try:
        print("⏱️  Generating fairness gap combined tables...")
        fairness_config = project_root / "configs/fairness_gap_combined.yaml"
        
        if fairness_config.exists():
            subprocess.run(
                [sys.executable, "-m", "src.benchmark.tables.fairness_gap_combined", str(fairness_config)],
                cwd=str(project_root),
                check=True
            )
            print("✅ Fairness gap tables generated successfully")
        else:
            print(f"⚠️  Config not found: {fairness_config}. Skipping fairness gap tables.")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Fairness gap tables generation failed: {e}")
    except Exception as e:
        print(f"⚠️  Error during fairness gap tables generation: {e}")
    
    # Step 2b: Generate initial performance combined table
    print()
    try:
        print("⏱️  Generating initial performance combined table...")
        perf_config = project_root / "configs/initial_performance_combined.yaml"
        
        if perf_config.exists():
            subprocess.run(
                [sys.executable, "-m", "src.benchmark.tables.initial_performance_combined", str(perf_config)],
                cwd=str(project_root),
                check=True
            )
            print("✅ Initial performance table generated successfully")
        else:
            print(f"⚠️  Config not found: {perf_config}. Skipping initial performance table.")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Initial performance table generation failed: {e}")
    except Exception as e:
        print(f"⚠️  Error during initial performance table generation: {e}")

    # Step 2c: Generate cohort tables for article inputs
    print()
    try:
        print("⏱️  Generating cohort tables...")
        for cohort_config in [
            project_root / "configs" / "folktables_benchmark_temporal.yaml",
            project_root / "configs" / "folktables_benchmark_temporal_employment.yaml",
        ]:
            if cohort_config.exists():
                subprocess.run(
                    [sys.executable, "-m", "src.benchmark.tables.cohort_table", str(cohort_config)],
                    cwd=str(project_root),
                    check=True,
                )
            else:
                print(f"⚠️  Config not found: {cohort_config}. Skipping cohort table.")
        print("✅ Cohort tables generated successfully")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Cohort table generation failed: {e}")
    except Exception as e:
        print(f"⚠️  Error during cohort table generation: {e}")
    
    # Step 3: Sync plots to paper
    print(f"\n{'='*80}")
    print("STEP 3: Sync Plots to Paper")
    print(f"{'='*80}")
    print()
    
    try:
        print("⏱️  Syncing latest plots to paper figures...")
        sync_script = project_root / "paper" / "sync_plots.py"
        
        if sync_script.exists():
            subprocess.run(
                [sys.executable, str(sync_script), "--task", "income"],
                cwd=str(project_root),
                check=True
            )
            print()
            subprocess.run(
                [sys.executable, str(sync_script), "--task", "employment"],
                cwd=str(project_root),
                check=True
            )
            print("✅ Plots synced successfully")
        else:
            print(f"⚠️  Sync script not found: {sync_script}. Skipping plot sync.")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Plot sync failed: {e}")
    except Exception as e:
        print(f"⚠️  Error during plot sync: {e}")
    
    # Step 4: Rebuild paper
    print(f"\n{'='*80}")
    print("STEP 4: Rebuild LaTeX Paper")
    print(f"{'='*80}")
    print()
    
    try:
        print("⏱️  Compiling LaTeX paper...")
        paper_dir = project_root / "paper"
        rebuild_script = paper_dir / "rebuild_paper.sh"
        
        if rebuild_script.exists():
            result = subprocess.run(
                ["/bin/bash", str(rebuild_script)],
                cwd=str(paper_dir),
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("✅ Paper compiled successfully")
                # Extract and show summary
                if "Elapsed time" in result.stdout:
                    for line in result.stdout.split('\n'):
                        if "pages" in line or "Elapsed" in line or "PDF size" in line:
                            print(f"   {line.strip()}")
            else:
                print(f"⚠️  Paper compilation had issues:")
                print(result.stdout)
        else:
            print(f"⚠️  Rebuild script not found: {rebuild_script}. Skipping paper rebuild.")
    except Exception as e:
        print(f"⚠️  Error during paper rebuild: {e}")
    
    # Final summary
    print(f"\n{'='*80}")
    print("OVERALL COMPLETION SUMMARY")
    print(f"{'='*80}")
    total_duration = time.time() - total_start
    print(f"✅ All benchmarks completed successfully")
    print(f"✅ LaTeX file updated with new run IDs")
    print(f"✅ Fairness gap tables generated")
    print(f"✅ Initial performance table generated")
    print(f"✅ Plots synced to paper")
    print(f"✅ Paper compiled")
    print()
    print(f"Total duration: {total_duration/3600:.2f} hours ({total_duration/60:.1f} minutes)")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print(f"Paper location: {project_root / 'paper' / 'articol.pdf'}")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code if exit_code is not None else 0)

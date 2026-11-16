# Benchmark-only third phase; executes after the tree/dense trainer has closed.
if __name__ == "__main__":
    import os as _benchmark_os
    import sys as _benchmark_sys
    from pathlib import Path as _BenchmarkPath

    _benchmark_run = _BenchmarkPath(_benchmark_sys.argv[_benchmark_sys.argv.index("--run-dir") + 1])
    if _benchmark_run.name == "tree":
        _benchmark_root = _BenchmarkPath(
            "/large_storage/goodarzilab/nikitamounier/assembly/acceleration-20261107-sparse"
        )
        _benchmark_os.chdir(_benchmark_root)
        _benchmark_python = str(_benchmark_root / ".venv/bin/python")
        _benchmark_os.execv(
            _benchmark_python, [_benchmark_python, "-m", "operational.acceleration-20261107.sparse_phase"]
        )

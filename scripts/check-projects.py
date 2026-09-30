"""Run isolated project suites; root pytest collection collides on test filenames.

Uses the current environment; install dependencies first. No cloud/model/data
integration is enabled. --notebooks additionally runs percent-format notebooks.
"""

import argparse
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebooks", action="store_true")
    parser.add_argument("--match", default="week-*")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    projects = [
        p for p in sorted((root / "projects").glob(args.match)) if (p / "pyproject.toml").exists()
    ]
    if not projects:
        parser.error("no matching projects")
    for project in projects:
        print(f"\n== {project.name} ==", flush=True)
        env = {
            **os.environ,
            "PYTHONPATH": str(project / "src"),
            "OMP_NUM_THREADS": "1",
            "MPLBACKEND": "Agg",
            "TOKENIZERS_PARALLELISM": "false",
        }
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "pytest.xml"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests",
                    "-o",
                    "addopts=-q",
                    f"--junitxml={report}",
                ],
                cwd=project,
                env=env,
                check=False,
            )
            if result.returncode:
                return result.returncode
            for suite in ET.parse(report).getroot().iter("testsuite"):
                for key in totals:
                    totals[key] += int(suite.get(key, 0))
        if args.notebooks:
            for notebook in sorted((project / "notebooks").glob("*.py")):
                result = subprocess.run(
                    [sys.executable, str(notebook)], cwd=project, env=env, check=False
                )
                if result.returncode:
                    return result.returncode
    print(f"\nPROJECT TOTAL: {len(projects)} packages; {totals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

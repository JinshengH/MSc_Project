from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


def resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def scratch_project_path(*parts: str, local: str | Path | None = None) -> Path:
    """Resolve a project path under $SCRATCH on HPC, or project root locally."""
    scratch = os.getenv("SCRATCH")
    if scratch:
        return Path(scratch) / "msc_project" / Path(*parts)
    if local is not None:
        return resolve_path(local)
    return PROJECT_ROOT / Path(*parts)


def output_dir(results_dir: str | Path, exp_name: str) -> Path:
    return resolve_path(results_dir) / exp_name

"""Route every derived artifact of a run into assets/results/<scene>/<timestamp>/"""


from pathlib import Path


# ========================================================================
# Roots
# ========================================================================
RESULTS_ROOT = Path(__file__).resolve().parents[1] / "assets" / "results"
PAPER_DIR = RESULTS_ROOT / "paper"


# ========================================================================
# Config parsing
# ========================================================================
def parse_dataset_path_from_config(config_path: Path) -> Path | None:
    """Read the dataset PosixPath block out of a nerfstudio config.yml."""
    lines = config_path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if line.strip() != "data: !!python/object/apply:pathlib.PosixPath":
            continue
        parts = []
        cursor = index + 1
        while cursor < len(lines) and lines[cursor].startswith("      - "):
            parts.append(lines[cursor].split("- ", 1)[1])
            cursor += 1
        if parts:
            return Path(*parts)
    return None


def scene_name(config_path: Path) -> str:
    """Return the scene folder name the run was trained on."""
    dataset_path = parse_dataset_path_from_config(config_path)
    if dataset_path is None:
        raise SystemExit(f"Could not find dataset path inside config: {config_path}")
    return dataset_path.name


# ========================================================================
# Artifact directories
# ========================================================================
def run_artifact_dir(config_path: Path, kind: str) -> Path:
    """Return assets/results/<scene>/<timestamp>/<kind>, creating it."""
    config_path = Path(config_path).resolve()
    directory = RESULTS_ROOT / scene_name(config_path) / config_path.parent.name / kind
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def paper_dir() -> Path:
    """Return assets/results/paper for cross-run figures, creating it."""
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    return PAPER_DIR

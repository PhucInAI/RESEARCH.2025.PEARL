"""Route every derived artifact of a run into assets/results/<scene>/<timestamp>/"""


import re
from pathlib import Path


# ========================================================================
# Roots
# ========================================================================
REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_ROOT = REPO_ROOT / "assets"
DATA_ROOT = ASSETS_ROOT / "data"
OUTPUTS_ROOT = ASSETS_ROOT / "outputs"
RESULTS_ROOT = ASSETS_ROOT / "results"
PAPER_DIR = RESULTS_ROOT / "paper"


# ========================================================================
# Config parsing
# ========================================================================
# Read as text rather than through yaml.load: this module is imported by plotting
# scripts that have no reason to pull in nerfstudio. A config carries three `data` keys
# — the trainer's, the datamanager's and the dataparser's — but `anchor_locations`
# resolves all of them to the same place, so the first one holding a path is the
# dataset. Quotes are stripped because PyYAML falls back to quoting a tagged scalar
# whose text cannot be written plain (`pearl.config_yaml`).
DATA_PATH_RE = re.compile(r"(?m)^\s*data: !path (?P<path>.+)$")


def parse_dataset_path_from_config(config_path: Path) -> Path | None:
    """Read the dataset path out of a PEARL config.yml, absolute."""
    match = DATA_PATH_RE.search(Path(config_path).read_text(encoding="utf-8"))
    return REPO_ROOT / match.group("path").strip("'\"") if match else None


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
    ts_dir = config_path.parent
    if ts_dir.name == "best":  # a best/ export sits one level below its run
        ts_dir = ts_dir.parent
    directory = RESULTS_ROOT / scene_name(config_path) / ts_dir.name / kind
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def paper_dir() -> Path:
    """Return assets/results/paper for cross-run figures, creating it."""
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    return PAPER_DIR

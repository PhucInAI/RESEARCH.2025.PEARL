"""Route every derived artifact of a run into assets/results/<scene>/<timestamp>/"""


from pathlib import Path
from typing import NamedTuple


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
# PosixPath blocks inside a saved config.yml
# ========================================================================
POSIXPATH_TAG = "!!python/object/apply:pathlib.PosixPath"


class PosixPathBlock(NamedTuple):
    """A `key: <PosixPath>` YAML block located inside a config's text."""

    start: int
    end: int
    indent: str
    parts: list[str]


def find_posixpath_blocks(text: str, key: str) -> list[PosixPathBlock]:
    """Locate every `key: <PosixPath>` block in a config's text, in file order."""
    lines = text.splitlines(keepends=True)
    blocks, start = [], 0
    while start < len(lines):
        if lines[start].strip() != f"{key}: {POSIXPATH_TAG}":
            start += 1
            continue
        line = lines[start]
        indent = line[: len(line) - len(line.lstrip())]
        parts, end = [], start + 1
        while end < len(lines) and lines[end].startswith(f"{indent}- "):
            parts.append(lines[end][len(indent) + 2 :].strip())
            end += 1
        blocks.append(PosixPathBlock(start, end, indent, parts))
        start = end
    return blocks


def find_posixpath_block(text: str, key: str) -> PosixPathBlock | None:
    """Locate the first `key: <PosixPath>` block in a config's text."""
    blocks = find_posixpath_blocks(text, key)
    return blocks[0] if blocks else None


def posixpath_block(key: str, path: Path, indent: str = "") -> str:
    """Render `key: <path>` as the PosixPath block nerfstudio writes."""
    lines = [f"{indent}{key}: {POSIXPATH_TAG}"]
    lines += [f"{indent}- {part}" for part in Path(path).resolve().parts]
    return "\n".join(lines) + "\n"


def replace_posixpath_block(text: str, key: str, block: PosixPathBlock, path: Path) -> str:
    """Point one located `key: <PosixPath>` block at `path`, indent preserved."""
    lines = text.splitlines(keepends=True)
    head = "".join(lines[: block.start])
    tail = "".join(lines[block.end :])
    return head + posixpath_block(key, path, block.indent) + tail


# ------------------------------------------------------------------
# Absolutising a saved config (KI-001)
# ------------------------------------------------------------------
# Runs saved before 2026-07 stored `output_dir: outputs` and `data: data/...`,
# which resolved only through the root symlinks that no longer exist. nerfstudio
# rebuilds the checkpoint dir from output_dir on every load (eval_utils.eval_setup
# overwrites load_dir), so these two blocks decide whether a run is loadable.
# Every `data` block is rewritten, not just the first: `ns-train --data` copies the
# path onto the datamanager as well (train.py), so a config can carry several.
REROOTED_BLOCKS = (
    ("output_dir", "outputs", OUTPUTS_ROOT),
    ("data", "data", DATA_ROOT),
)


def _reroot(parts: list[str], anchor: str, root: Path) -> Path:
    """Re-anchor a stored path onto `root`, keeping everything after `anchor`."""
    tail = parts[len(parts) - parts[::-1].index(anchor) :]
    return root.joinpath(*tail)


def absolutize_config_text(text: str) -> str:
    """Rewrite every output_dir and dataset block of a config to an absolute path."""
    for key, anchor, root in REROOTED_BLOCKS:
        # Back to front: splicing a block shifts the line numbers of later ones.
        for block in reversed(find_posixpath_blocks(text, key)):
            text = replace_posixpath_block(text, key, block, _reroot(block.parts, anchor, root))
    return text


# ========================================================================
# Config parsing
# ========================================================================
def parse_dataset_path_from_config(config_path: Path) -> Path | None:
    """Read the dataset PosixPath block out of a nerfstudio config.yml."""
    block = find_posixpath_block(Path(config_path).read_text(encoding="utf-8"), "data")
    return Path(*block.parts) if block and block.parts else None


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

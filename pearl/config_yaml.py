"""Dump and load nerfstudio configs under short tags and project-root-relative paths"""


import enum
import importlib
import types
from pathlib import Path, PurePath

import yaml
from nerfstudio.configs.base_config import PrintableConfig

from pearl.run_paths import REPO_ROOT


# ========================================================================
# Tags
# ========================================================================
# A config class and an enum carry their own bare name (`!AdamOptimizerConfig`,
# `!EventName`); everything else needs a fixed tag. `!path` is a location and is
# stored relative to REPO_ROOT whenever it sits inside the repo, so a config never
# hard-codes this machine. `!relpath` is a fragment interpreted against some other
# directory (`nerfstudio_models` under the run dir, `sparse`/`images` under the
# dataset) and is stored and loaded verbatim.
TAG_PATH = "!path"
TAG_RELPATH = "!relpath"
TAG_TYPE = "!type"
TAG_TUPLE = "!tuple"


# ========================================================================
# Dumping
# ========================================================================
class PearlDumper(yaml.Dumper):
    """yaml.Dumper that writes configs under short tags and repo-relative paths."""

    def ignore_aliases(self, data):
        """Never emit &id001 / *id002 — a config tree is acyclic, so they only add noise."""
        return True

    def choose_scalar_style(self):
        """Leave a PEARL-tagged scalar unquoted — its explicit tag already fixes its type."""
        # PyYAML quotes any scalar whose tag it had to write out, which is every one
        # of ours. Only the `!`-prefixed tags are relaxed; a `tag:yaml.org,2002:` one
        # may be relying on those quotes to stay a string.
        style = super().choose_scalar_style()
        if self.event.tag.startswith("!") and self.analysis.allow_block_plain:
            return ""
        return style


def _represent_config(dumper: yaml.Dumper, config: PrintableConfig) -> yaml.Node:
    """Write a config dataclass as `!<ClassName>` over its fields."""
    return dumper.represent_mapping(f"!{type(config).__name__}", config.__dict__)


def _represent_path(dumper: yaml.Dumper, path: PurePath) -> yaml.Node:
    """Write a location repo-relative when it is inside the repo, a fragment verbatim."""
    if not path.is_absolute():
        return dumper.represent_scalar(TAG_RELPATH, str(path))
    if path.is_relative_to(REPO_ROOT):
        return dumper.represent_scalar(TAG_PATH, str(path.relative_to(REPO_ROOT)))
    return dumper.represent_scalar(TAG_PATH, str(path))


def _represent_type(dumper: yaml.Dumper, target) -> yaml.Node:
    """Write a `_target` class or function as its dotted import path."""
    return dumper.represent_scalar(TAG_TYPE, f"{target.__module__}.{target.__qualname__}")


def _represent_enum(dumper: yaml.Dumper, member: enum.Enum) -> yaml.Node:
    """Write an enum member as `!<EnumName> <value>`."""
    return dumper.represent_scalar(f"!{type(member).__name__}", str(member.value))


def _represent_tuple(dumper: yaml.Dumper, values: tuple) -> yaml.Node:
    """Write a tuple as a tagged sequence so it does not come back as a list."""
    return dumper.represent_sequence(TAG_TUPLE, values)


PearlDumper.add_multi_representer(PrintableConfig, _represent_config)
PearlDumper.add_multi_representer(PurePath, _represent_path)
PearlDumper.add_multi_representer(type, _represent_type)
PearlDumper.add_multi_representer(types.FunctionType, _represent_type)
PearlDumper.add_multi_representer(enum.Enum, _represent_enum)
PearlDumper.add_representer(tuple, _represent_tuple)


def dump_config_text(config) -> str:
    """Serialise a nerfstudio config to the PEARL config.yml format."""
    return yaml.dump(config, Dumper=PearlDumper)


# ========================================================================
# Loading
# ========================================================================
# A short tag names no module, so resolution is by bare class name over the live
# subclass tree. Importing nerfstudio.configs.method_configs is what fills that tree:
# it pulls in every built-in method's config classes and, through plugin discovery,
# every installed plugin's — PEARL's included. ns-train and ns-eval have already done
# it before they read a config; a bare `import yaml` script has not, so a first miss
# does it rather than failing on a class that was only ever a module away.
CONFIG_REGISTRY_MODULE = "nerfstudio.configs.method_configs"


def _subclasses(root: type):
    """Yield every subclass of `root`, transitively."""
    for subclass in root.__subclasses__():
        yield subclass
        yield from _subclasses(subclass)


def _resolve_name(root: type, name: str) -> type:
    """Find the one subclass of `root` called `name`."""
    matches = [c for c in _subclasses(root) if c.__name__ == name]
    if not matches:
        importlib.import_module(CONFIG_REGISTRY_MODULE)
        matches = [c for c in _subclasses(root) if c.__name__ == name]
    assert len(matches) == 1, f"Expected exactly one {root.__name__} named {name}, found {matches}"
    return matches[0]


def _construct_named(loader: yaml.Loader, suffix: str, node: yaml.Node):
    """Rebuild whatever `!<ClassName>` names — a mapping is a config, a scalar an enum."""
    if isinstance(node, yaml.ScalarNode):
        return _resolve_name(enum.Enum, suffix)(loader.construct_scalar(node))
    # Bypassing __init__ is how PyYAML restores a plain object, and is what lets a
    # config whose defaults have since changed load back exactly as it was saved.
    config = object.__new__(_resolve_name(PrintableConfig, suffix))
    config.__dict__.update(loader.construct_mapping(node, deep=True))
    return config


def _construct_path(loader: yaml.Loader, node: yaml.Node) -> Path:
    """Re-anchor a stored location on REPO_ROOT, so it resolves from any cwd."""
    return REPO_ROOT / Path(loader.construct_scalar(node))


def _construct_relpath(loader: yaml.Loader, node: yaml.Node) -> Path:
    """Rebuild a path fragment verbatim — its base is another directory, not the repo."""
    return Path(loader.construct_scalar(node))


def _construct_type(loader: yaml.Loader, node: yaml.Node):
    """Import the class or function named by a dotted `_target` path."""
    module, _, attribute = loader.construct_scalar(node).rpartition(".")
    return getattr(importlib.import_module(module), attribute)


def _construct_tuple(loader: yaml.Loader, node: yaml.Node) -> tuple:
    """Rebuild a tuple from a tagged sequence."""
    return tuple(loader.construct_sequence(node, deep=True))


# --------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------
# Registered on the stock loaders rather than a PearlLoader subclass: nerfstudio's
# eval_utils.eval_setup calls yaml.load(..., Loader=yaml.Loader) and cannot be told
# otherwise. Purely additive — these are new tags, so every config already on disk
# keeps loading, and `!!python/object:` stays the fallback for anything a
# representer above does not cover. The `!` multi-constructor claims that whole tag
# prefix process-wide, but an exact-tag constructor still wins over it, so only
# another library's own `!`-prefix *multi*-constructor could be shadowed.
def register_yaml_handlers() -> None:
    """Teach the stock PyYAML loaders every tag PearlDumper writes."""
    for loader in (yaml.Loader, yaml.FullLoader, yaml.UnsafeLoader):
        yaml.add_multi_constructor("!", _construct_named, Loader=loader)
        yaml.add_constructor(TAG_PATH, _construct_path, Loader=loader)
        yaml.add_constructor(TAG_RELPATH, _construct_relpath, Loader=loader)
        yaml.add_constructor(TAG_TYPE, _construct_type, Loader=loader)
        yaml.add_constructor(TAG_TUPLE, _construct_tuple, Loader=loader)


# ========================================================================
# Location anchoring
# ========================================================================
# `ns-train --data assets/data/<scene>` stores whatever the shell passed, so a run
# launched from the repo root would otherwise save a cwd-dependent fragment. These
# two fields are the ones that decide loadability: nerfstudio rebuilds the
# checkpoint dir from output_dir on every load (eval_utils.eval_setup overwrites
# load_dir), and the dataparser reopens the dataset from `data`. `--data` is copied
# onto the datamanager as well (train.py), so a config carries several.
def dataset_owners(config) -> tuple:
    """Return every object of a config that carries a `data` path."""
    datamanager = config.pipeline.datamanager
    return (config, datamanager, datamanager.dataparser)


def anchor_locations(config) -> None:
    """Resolve output_dir and every dataset path of a config to an absolute path."""
    config.output_dir = Path(config.output_dir).resolve()
    for owner in dataset_owners(config):
        if owner.data is not None:
            owner.data = Path(owner.data).resolve()

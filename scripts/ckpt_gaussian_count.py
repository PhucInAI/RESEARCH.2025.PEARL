"""Best-effort gaussian count from a nerfstudio .ckpt, stdlib-only (no torch).

A torch checkpoint is a zip whose pickle stream rebuilds tensors through
torch._utils._rebuild_tensor_v2 with an explicit size tuple. Unpickling
with stubbed torch globals therefore yields every parameter's SHAPE
without touching tensor data — enough to read N off the means tensor on
a machine with no python environment (KI-003). The torch-based cluster
reporters (ws_report.py / efficiency_report.py) remain the source of
truth; treat this as a preview number.

    python3 scripts/ckpt_gaussian_count.py <path/to/step-*.ckpt|best.ckpt>
"""


import pickle
import sys
import zipfile
from collections import OrderedDict


# ========================================================================
# Stubbed unpickling
# ========================================================================
class _Any:
    """Permissive stand-in for any global the checkpoint references."""

    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return None

    def __setstate__(self, state):
        pass


def _rebuild_tensor_stub(storage, offset, size, stride, *a, **k):
    """Keep only the shape a tensor is rebuilt with."""
    return ("TENSOR", tuple(size))


class StubUnpickler(pickle.Unpickler):
    """Unpickle a torch checkpoint with every torch/numpy global stubbed."""

    def find_class(self, module, name):
        if name in ("_rebuild_tensor_v2", "_rebuild_tensor", "_rebuild_parameter"):
            return _rebuild_tensor_stub
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        return _Any

    def persistent_load(self, pid):
        return None


# ========================================================================
# Shape harvesting
# ========================================================================
def collect_shapes(obj, path="", out=None):
    """Walk the unpickled object and collect (key path, shape) per tensor."""
    if out is None:
        out = []
    if isinstance(obj, tuple) and len(obj) == 2 and obj[0] == "TENSOR":
        out.append((path, obj[1]))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            collect_shapes(v, f"{path}.{k}" if path else str(k), out)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            collect_shapes(v, f"{path}[{i}]", out)
    return out


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    ckpt = sys.argv[1]

    with zipfile.ZipFile(ckpt) as zf:
        pkl = next(n for n in zf.namelist() if n.endswith("data.pkl"))
        with zf.open(pkl) as f:
            obj = StubUnpickler(f).load()

    shapes = collect_shapes(obj)
    means = [(p, s) for p, s in shapes
             if "means" in p.lower() and "optimizers" not in p.lower() and len(s) == 2 and s[1] == 3]
    if not means:
        print("No means-like tensor found; all 2D tensors ending in 3:")
        for p, s in shapes:
            if len(s) == 2 and s[1] == 3:
                print(f"  {p}: {s}")
        raise SystemExit(1)

    for p, s in means:
        print(f"{p}: shape {s}")
    n = max(s[0] for _, s in means)
    print(f"gaussian_count: {n}")


if __name__ == "__main__":
    main()

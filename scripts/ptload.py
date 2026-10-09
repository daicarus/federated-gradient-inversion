"""Read a PyTorch .pt checkpoint into numpy arrays without torch installed.

Handles the zipfile-based format written by torch.save in 1.6+. Useful for
inspecting results/ in an environment without torch.

    from ptload import load
    d = load("results/med_recons.pt")
    d["all_recon"]["none|IG"].shape   # (10, 3, 32, 32)
"""
import zipfile
import pickle
import collections
import numpy as np

_DT = {
    "FloatStorage": np.float32,
    "DoubleStorage": np.float64,
    "HalfStorage": np.float16,
    "LongStorage": np.int64,
    "IntStorage": np.int32,
    "ByteStorage": np.uint8,
    "BoolStorage": np.bool_,
    "BFloat16Storage": np.uint16,
}


class _Storage:
    def __init__(self, name):
        self.name = name


def load(path):
    z = zipfile.ZipFile(path)
    prefix = z.namelist()[0].split("/")[0]
    cache = {}

    def sb(key, dtype):
        if key not in cache:
            cache[key] = np.frombuffer(z.read(f"{prefix}/data/{key}"), dtype=dtype)
        return cache[key]

    def rebuild(storage, offset, size, stride, *a):
        if len(size) == 0:
            return storage[offset]
        it = storage.itemsize
        return np.lib.stride_tricks.as_strided(
            storage[offset:],
            shape=tuple(size),
            strides=tuple(s * it for s in stride),
        ).copy()

    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if name == "_rebuild_tensor_v2":
                return rebuild
            if name == "_rebuild_parameter":
                return lambda data, *a: data
            if mod == "collections" and name == "OrderedDict":
                return collections.OrderedDict
            if name.endswith("Storage"):
                return _Storage(name)
            if mod.startswith("torch"):
                return lambda *a, **k: None
            return super().find_class(mod, name)

        def persistent_load(self, pid):
            _, st, key, loc, numel = pid
            return sb(key, _DT.get(getattr(st, "name", str(st)), np.float32))

    return U(z.open(f"{prefix}/data.pkl")).load()


if __name__ == "__main__":
    import sys

    for p in sys.argv[1:]:
        d = load(p)
        print(p)
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, dict):
                    print(f"  {k}: dict with {len(v)} keys")
                    for kk, vv in list(v.items())[:3]:
                        print(f"    {kk}: {getattr(vv, 'shape', type(vv).__name__)}")
                else:
                    print(f"  {k}: {getattr(v, 'shape', type(v).__name__)}")

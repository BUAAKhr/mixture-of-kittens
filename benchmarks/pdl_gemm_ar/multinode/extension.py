from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_pdl_extension() -> ModuleType:
    benchmark_root = Path(__file__).resolve().parents[1]
    candidates = sorted(benchmark_root.glob("_C*.so"))
    if len(candidates) != 1:
        raise ImportError(
            "expected exactly one benchmarks/pdl_gemm_ar/_C*.so, found "
            f"{len(candidates)}"
        )
    extension_path = candidates[0].resolve()
    loaded = sys.modules.get("_C")
    if loaded is not None:
        loaded_path = Path(str(getattr(loaded, "__file__", ""))).resolve()
        if loaded_path != extension_path:
            raise ImportError(
                f"_C is already loaded from {loaded_path}, expected "
                f"{extension_path}"
            )
        return loaded
    spec = importlib.util.spec_from_file_location("_C", extension_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create an import spec for {extension_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_C"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop("_C", None)
        raise
    return module

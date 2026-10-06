"""Load an engine once so its models are downloaded into the image."""

import importlib.util
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
engine = sys.argv[1]
spec = importlib.util.spec_from_file_location("adapter", root / "engines" / engine / "adapter.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.create("cpu").load()
print(f"{engine}: models ready")

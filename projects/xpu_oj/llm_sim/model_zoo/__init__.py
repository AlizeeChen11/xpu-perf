from pathlib import Path

from .seed_oss import model_configs as seed_oss_configs
from .qwen3_dense import model_configs as qwen3_dense_configs
from .qwen3_moe import model_configs as qwen3_moe_configs


BASE_MODEL_MAPPING = {
    "seed_oss": seed_oss_configs,
    "qwen3_dense": qwen3_dense_configs, 
    "qwen3_moe": qwen3_moe_configs,
}


def _load_version() -> str:
    version_file = Path(__file__).with_name("VERSION")
    try:
        return version_file.read_text(encoding="utf-8").strip()
    except OSError:
        return "0.0.0"


__version__ = _load_version()

__all__ = ["BASE_MODEL_MAPPING", "__version__"]


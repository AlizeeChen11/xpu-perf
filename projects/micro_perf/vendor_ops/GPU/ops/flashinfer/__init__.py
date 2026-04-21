import importlib.metadata
from xpu_perf.micro_perf.core.op import ProviderRegistry

try:
    ProviderRegistry.register_provider_info("flashinfer", {
        "flashinfer": importlib.metadata.version("flashinfer-python")
    })
except Exception:
    pass

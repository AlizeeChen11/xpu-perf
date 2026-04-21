import pathlib
import torch

from xpu_perf.micro_perf.core.op import ProviderRegistry


@ProviderRegistry.register_vendor_impl("gemm", "torch")
class GPUGemmOp:
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def vendor_parser(self):
        super().vendor_parser()
        if self.dtype == "float32":
            torch.set_float32_matmul_precision("highest")
        elif self.dtype == "tfloat32":
            torch.set_float32_matmul_precision("high")

    def __del__(self):
        torch.set_float32_matmul_precision("highest")
        getattr(super(), "__del__", lambda: None)()

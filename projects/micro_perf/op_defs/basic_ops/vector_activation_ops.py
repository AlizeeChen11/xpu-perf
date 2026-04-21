import pathlib
import torch
from functools import partial

from xpu_perf.micro_perf.core.utils import OpTensorInfo, calc_tensor_size
from xpu_perf.micro_perf.core.op import BasicOp, ProviderRegistry
from .vector_sfu_ops import CosOp


@ProviderRegistry.register_base_impl("gelu", "ComputeEngine")
class GeluOp(CosOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=False
        )
    def vendor_impl_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = torch.nn.functional.gelu(src)
        return dst



@ProviderRegistry.register_base_impl("silu", "ComputeEngine")
class SiluOp(CosOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=False
        )
    def vendor_impl_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = torch.nn.functional.silu(src)
        return dst


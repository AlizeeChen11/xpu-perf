import pathlib
import torch
from functools import partial

from xpu_perf.micro_perf.core.utils import OpTensorInfo, calc_tensor_size
from xpu_perf.micro_perf.core.op import BasicOp, ProviderRegistry
from .vector_linear_ops import AddOp







@ProviderRegistry.register_base_impl("div", "ComputeEngine")
class DivOp(AddOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def vendor_impl_run(self, tensor_mapping):
        a = tensor_mapping["a"]
        b = tensor_mapping["b"]
        c = tensor_mapping["c"]
        c = torch.div(a, b, out=c)
        return c
    

@ProviderRegistry.register_base_impl("cos", "ComputeEngine")
class CosOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare_args(self):
        self.dtype = self.args_dict["dtype"]
        self.torch_dtype = getattr(torch, self.dtype)

        self.batch_size = self.args_dict["batch_size"]
        self.dim_size = self.args_dict["dim_size"]

        self.input_tensor_info = {
            "src": OpTensorInfo(
                shape=[self.batch_size, self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            )
        }
        self.output_tensor_info = {
            "dst": OpTensorInfo(
                shape=[self.batch_size, self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            )
        }
        
        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size
        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes
        self.algo_size = 0
        self.bus_size = 0
        self.calc_flops = self.batch_size * self.dim_size

    def vendor_impl_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = tensor_mapping["dst"]
        torch.cos(src, out=dst)
        return dst
    

@ProviderRegistry.register_base_impl("sin", "ComputeEngine")
class SinOp(CosOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def vendor_impl_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = tensor_mapping["dst"]
        torch.sin(src, out=dst)
        return dst
    

@ProviderRegistry.register_base_impl("exp", "ComputeEngine")
class ExpOp(CosOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def vendor_impl_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = tensor_mapping["dst"]
        torch.exp(src, out=dst)
        return dst


@ProviderRegistry.register_base_impl("log", "ComputeEngine")
class LogOp(CosOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def vendor_impl_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = tensor_mapping["dst"]
        torch.log(src, out=dst)
        return dst


@ProviderRegistry.register_base_impl("sqrt", "ComputeEngine")
class SqrtOp(CosOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def vendor_impl_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = tensor_mapping["dst"]
        torch.sqrt(src, out=dst)
        return dst


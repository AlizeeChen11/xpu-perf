import pathlib
import torch
from functools import partial

from xpu_perf.micro_perf.core.utils import OpTensorInfo, calc_tensor_size
from xpu_perf.micro_perf.core.op import BasicOp, ProviderRegistry







@ProviderRegistry.register_base_impl("add", "ComputeEngine")
class AddOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare_args(self):
        self.dtype = self.args_dict["dtype"]
        self.torch_dtype = getattr(torch, self.dtype)
        
        self.batch_size = self.args_dict["batch_size"]
        self.dim_size = self.args_dict["dim_size"]

        self.input_tensor_info = {
            "a": OpTensorInfo(
                shape=[self.batch_size, self.dim_size],
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(), 
            ), 
            "b": OpTensorInfo(
                shape=[self.batch_size, self.dim_size],
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(), 
            )
        }
        self.output_tensor_info = {
            "c": OpTensorInfo(
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
        a = tensor_mapping["a"]
        b = tensor_mapping["b"]
        c = tensor_mapping["c"]
        c = torch.add(a, b, out=c)
        return c

@ProviderRegistry.register_base_impl("sub", "ComputeEngine")
class SubOp(AddOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
    def vendor_impl_run(self, tensor_mapping):
        a = tensor_mapping["a"]
        b = tensor_mapping["b"]
        c = tensor_mapping["c"]
        c = torch.sub(a, b, out=c)
        return c

@ProviderRegistry.register_base_impl("mul", "ComputeEngine")
class MulOp(AddOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
    def vendor_impl_run(self, tensor_mapping):
        a = tensor_mapping["a"]
        b = tensor_mapping["b"]
        c = tensor_mapping["c"]
        c = torch.mul(a, b, out=c)
        return c

@ProviderRegistry.register_base_impl("cast", "ComputeEngine")
class CastOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare_args(self):
        self.dtype = self.args_dict["dtype"]
        # support:
        # fp32: fp32 --> bf16
        # fp16: fp16 --> fp32
        # bf16: bf16 --> fp32
        if self.dtype == "float32":
            self.src_dtype = torch.float32
            self.dst_dtype = torch.bfloat16
        elif self.dtype == "float16":
            self.src_dtype = torch.float16
            self.dst_dtype = torch.float32
        elif self.dtype == "bfloat16":
            self.src_dtype = torch.bfloat16
            self.dst_dtype = torch.float32
        else:
            raise NotImplementedError
                
        self.batch_size = self.args_dict["batch_size"]
        self.dim_size = self.args_dict["dim_size"]

        self.input_tensor_info = {
            "src": OpTensorInfo(
                shape=[self.batch_size, self.dim_size],
                dtype=self.src_dtype,
                device=self.backend.get_torch_device_name(),
            )
        }
        self.output_tensor_info = {
            "dst": OpTensorInfo(
                shape=[self.batch_size, self.dim_size],
                dtype=self.dst_dtype,
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

        self._create_tensors_func = partial(
            self._create_in_out_tensors, 
            create_inputs=True,
            create_outputs=False
        )

    def vendor_impl_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = src.to(dtype=self.dst_dtype)
        return dst



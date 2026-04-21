"""LLM op: swiglu (base implementation)."""
from ._common import *

@ProviderRegistry.register_base_impl("swiglu", "ComputeEngine")
class SwigluOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare_args(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise ValueError(
                f"SwigluOp only support llm arg_type, but got {self.arg_type}"
            )

        # predefined attrs
        self.sp_size = self.args_dict.get("sp_size", 1)
        self.num_tokens = self.args_dict["num_tokens"] // self.sp_size
        self.hidden_size = self.args_dict["hidden_size"]

        # 以下参数决定当前 swiglu 的具体数据类型
        self.dtype = self.args_dict["dtype"]

    def vendor_parser(self):
        if self.dtype in ["float32", "float16", "bfloat16"]:
            pass
        else:
            raise ValueError(f"dtype {self.dtype} not supported")

    def vendor_impl(self):
        self.torch_dtype = get_torch_dtype(self.dtype)

        # input/output tensors
        self.input_tensor_info = {
            "hidden_states": OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size * 2], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(),
            )
        }
        self.output_tensor_info = {
            "output_tokens": OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(),
            )
        }

        # calculator
        self.input_tensor_size = sum([
            calc_tensor_size(info) for info in self.input_tensor_info.values()
        ])
        self.output_tensor_size = sum([
            calc_tensor_size(info) for info in self.output_tensor_info.values()
        ])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        # creator func
        self._create_tensors_func = partial(
            self._create_in_out_tensors, 
            create_inputs=True, 
            create_outputs=False
        )
        self._run_func = self.vendor_impl_run

    def vendor_impl_run(self, tensor_mapping):
        hidden_states = tensor_mapping["hidden_states"]

        x1, x2 = torch.chunk(hidden_states, 2, dim=-1)
        output_tokens = torch.mul(torch.nn.functional.silu(x1), x2)

        return output_tokens



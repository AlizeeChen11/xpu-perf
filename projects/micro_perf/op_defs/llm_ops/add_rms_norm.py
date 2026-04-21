"""LLM op: add_rms_norm (base implementation)."""
from ._common import *

@ProviderRegistry.register_base_impl("add_rms_norm", "ComputeEngine")
class AddRmsNormOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare_args(self):
        self.arg_type = self.args_dict["arg_type"]
        if self.arg_type not in ["llm"]:
            raise ValueError(
                f"AddRmsNormOp only supports llm arg_type, got {self.arg_type}"
            )

        self.dtype = self.args_dict["dtype"]

        self.sp_size = self.args_dict.get("sp_size", 1)
        self.num_tokens = self.args_dict["num_tokens"] // self.sp_size
        self.hidden_size = self.args_dict["hidden_size"]

        self.eps = 1e-5

    def vendor_parser(self):
        if self.dtype != "bfloat16":
            raise ValueError(
                f"AddRmsNormOp only supports bfloat16 dtype, got {self.dtype}"
            )

    def vendor_impl(self):
        self.torch_dtype = get_torch_dtype(self.dtype)

        self.input_tensor_info = {
            "hidden_states": OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
            "residual": OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
            "norm_weight": OpTensorInfo(
                shape=[self.hidden_size, ],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.ones
            ),
        }
        self.output_tensor_info = {
            "after_res": OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
            "output": OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
        }

        self.input_tensor_size = sum(
            calc_tensor_size(info) for info in self.input_tensor_info.values()
        )
        self.output_tensor_size = sum(
            calc_tensor_size(info) for info in self.output_tensor_info.values()
        )
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=False
        )
        self._run_func = self.vendor_impl_run

    def vendor_impl_run(self, tensor_mapping):
        hidden_states = tensor_mapping["hidden_states"]
        residual = tensor_mapping["residual"]
        norm_weight = tensor_mapping["norm_weight"]

        output = torch.nn.functional.rms_norm(
            hidden_states + residual,
            normalized_shape=hidden_states.shape[-1:],
            weight=norm_weight,
            eps=self.eps
        )

        return output


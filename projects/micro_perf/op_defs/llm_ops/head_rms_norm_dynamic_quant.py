"""LLM op: head_rms_norm_dynamic_quant (base implementation)."""
from ._common import *

@ProviderRegistry.register_base_impl("head_rms_norm_dynamic_quant", "ComputeEngine")
class HeadRMSNormDynamicQuantOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare_args(self):
        self.arg_type = self.args_dict["arg_type"]
        if self.arg_type not in ["llm"]:
            raise ValueError(
                f"HeadRMSNormDynamicQuantOp only supports llm arg_type, got {self.arg_type}"
            )

        self.dtype = self.args_dict["dtype"]
        self.dst_dtype = self.args_dict["dst_dtype"]

        self.sp_size = self.args_dict.get("sp_size", 1)
        self.num_tokens = self.args_dict["num_tokens"] // self.sp_size
        self.head_num = self.args_dict["head_num"]
        self.head_dim = self.args_dict["head_dim"]

        self.eps = 1e-5

    def vendor_parser(self):
        if self.dtype != "bfloat16":
            raise ValueError(
                f"HeadRMSNormDynamicQuantOp only supports bfloat16 dtype, got {self.dtype}"
            )
        if self.dst_dtype not in ["int8", "float8"]:
            raise ValueError(
                f"HeadRMSNormDynamicQuantOp only supports dst_dtype int8 or float8, "
                f"got {self.dst_dtype}"
            )

    def vendor_impl(self):
        self.torch_dtype = get_torch_dtype(self.dtype)
        self.dst_torch_dtype = get_torch_dtype(self.dst_dtype)

        self.input_tensor_info = {
            "token_data": OpTensorInfo(
                shape=[self.num_tokens, self.head_num, self.head_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
            "norm_weight": OpTensorInfo(
                shape=[self.head_dim, ],
                dtype=torch.float32,
                device=self.backend.get_torch_device_name(),
                creator=torch.ones
            ),
            "smooth_scale": OpTensorInfo(
                shape=[self.head_num * self.head_dim],
                dtype=torch.float32,
                device=self.backend.get_torch_device_name(),
                creator=torch.ones
            ),
        }
        self.output_tensor_info = {
            "quant_tokens": OpTensorInfo(
                shape=[self.num_tokens, self.head_num * self.head_dim],
                dtype=self.dst_torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
            "per_token_scale": OpTensorInfo(
                shape=[self.num_tokens],
                dtype=torch.float32,
                device=self.backend.get_torch_device_name()
            )
        }

        self.input_tensor_size = 2 * sum(
            calc_tensor_size(info) for info in self.input_tensor_info.values()
        )
        self.output_tensor_size = sum(
            calc_tensor_size(info) for info in self.output_tensor_info.values()
        )
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = sum(
            calc_tensor_size(info) for info in self.input_tensor_info.values()
        )
        self.write_bytes = sum(
            calc_tensor_size(info) for info in self.output_tensor_info.values()
        )
        self.io_bytes = self.read_bytes + self.write_bytes

        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=False
        )
        self._run_func = self.vendor_impl_run

    def vendor_impl_run(self, tensor_mapping):
        token_data = tensor_mapping["token_data"]
        norm_weight = tensor_mapping["norm_weight"]
        smooth_scale = tensor_mapping["smooth_scale"]

        after_norm = torch.nn.functional.rms_norm(
            token_data,
            normalized_shape=token_data.shape[-1:],
            weight=norm_weight,
            eps=self.eps
        )
        after_norm = after_norm.view(self.num_tokens, self.head_num * self.head_dim)

        quant_tokens, per_token_scale = smooth_per_token_dynamic_quant(
            after_norm, smooth_scale, self.dst_torch_dtype
        )

        return quant_tokens, per_token_scale



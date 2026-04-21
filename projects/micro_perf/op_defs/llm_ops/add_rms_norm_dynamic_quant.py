"""LLM op: add_rms_norm_dynamic_quant (base implementation)."""
from ._common import *

@ProviderRegistry.register_base_impl("add_rms_norm_dynamic_quant", "ComputeEngine")
class AddRmsNormDynamicQuantOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare_args(self):
        self.arg_type = self.args_dict["arg_type"]
        if self.arg_type not in ["llm"]:
            raise ValueError(
                f"AddRmsNormDynamicQuantOp only supports llm arg_type, got {self.arg_type}"
            )

        self.dtype = self.args_dict["dtype"]
        self.dst_dtype = self.args_dict["dst_dtype"]

        self.sp_size = self.args_dict.get("sp_size", 1)
        self.num_tokens = self.args_dict["num_tokens"] // self.sp_size
        self.hidden_size = self.args_dict["hidden_size"]
        self.add_residual = self.args_dict.get("add_residual", True)
        self.output_mode = self.args_dict.get("output_mode", "none")

        self.eps = 1e-5

    def vendor_parser(self):
        if self.dtype != "bfloat16":
            raise ValueError(
                f"AddRmsNormDynamicQuantOp only supports bfloat16 dtype, got {self.dtype}"
            )
        if self.dst_dtype != "int8":
            raise ValueError(
                f"AddRmsNormDynamicQuantOp only supports dst_dtype int8, got {self.dst_dtype}"
            )
        if self.output_mode not in ["none", "res", "norm"]:
            raise ValueError(
                f"AddRmsNormDynamicQuantOp output_mode must be none, res, or norm, "
                f"got {self.output_mode}"
            )

    def vendor_impl(self):
        self.torch_dtype = get_torch_dtype(self.dtype)
        self.dst_torch_dtype = get_torch_dtype(self.dst_dtype)

        self.input_tensor_info = {
            "hidden_states": OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
            "norm_weight": OpTensorInfo(
                shape=[self.hidden_size, ],
                dtype=torch.float32,
                device=self.backend.get_torch_device_name(),
                creator=torch.ones
            ),
            "smooth_scale": OpTensorInfo(
                shape=[self.hidden_size],
                dtype=torch.float32,
                device=self.backend.get_torch_device_name(),
                creator=torch.ones
            )
        }
        if self.add_residual:
            self.input_tensor_info["residual"] = OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            )

        self.output_tensor_info = {
            "quant_tokens": OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size],
                dtype=self.dst_torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
            "per_token_scale": OpTensorInfo(
                shape=[self.num_tokens],
                dtype=torch.float32,
                device=self.backend.get_torch_device_name()
            )
        }
        if self.output_mode == "res":
            self.output_tensor_info["after_res"] = OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            )
        elif self.output_mode == "norm":
            self.output_tensor_info["after_norm"] = OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            )

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
        residual = tensor_mapping.get("residual", None)
        norm_weight = tensor_mapping["norm_weight"]
        smooth_scale = tensor_mapping["smooth_scale"]

        after_res = hidden_states
        if residual is not None:
            after_res = hidden_states + residual

        after_norm = torch.nn.functional.rms_norm(
            after_res,
            weight=norm_weight,
            normalized_shape=after_res.shape[-1:],
            eps=self.eps
        )

        quant_tokens, per_token_scale = smooth_per_token_dynamic_quant(
            after_norm, smooth_scale, self.dst_torch_dtype
        )

        if self.output_mode == "none":
            return quant_tokens, per_token_scale
        if self.output_mode == "res":
            return quant_tokens, per_token_scale, after_res
        return quant_tokens, per_token_scale, after_norm







"""
******************************************
Attention & rope & kvcache 算子
******************************************
"""


"""LLM op: moe_gather (base implementation)."""
from ._common import *

@ProviderRegistry.register_base_impl("moe_gather", "ComputeEngine")
class MoeGatherOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare_args(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise ValueError(
                f"MoeGatherOp only support llm arg_type, but got {self.arg_type}"
            )

        # predefined attrs
        self.num_tokens = self.args_dict["num_tokens"]
        self.hidden_size = self.args_dict["hidden_size"]

        # moe info
        self.num_experts = self.args_dict["num_experts"]
        self.topk = self.args_dict["topk"]

        # parallel info
        self.ep_size = self.args_dict.get("ep_size", 1)
        self.ep_rank = self.args_dict.get("ep_rank", 0)

        # residual info
        self.sp_size = self.args_dict.get("sp_size", 1)
        self.sp_rank = self.args_dict.get("sp_rank", 0)
        self.res_scale = self.args_dict.get("res_scale", 1.0)

        if self.sp_size > 1:
            self.num_res_tokens_per_rank = (self.num_tokens + self.sp_size - 1) // self.sp_size
            self.res_token_start = self.sp_rank * self.num_res_tokens_per_rank
            self.res_token_end = min(self.res_token_start + self.num_res_tokens_per_rank, self.num_tokens)
        else:
            self.num_res_tokens_per_rank = 1
            self.res_token_start = 0
            self.res_token_end = self.num_tokens


        # get moe token disptch info
        self.num_scatter_tokens, \
        self.num_scatter_tokens_per_rank, \
        self.num_experts_per_rank, \
        self.experts_start_idx, \
        self.experts_end_idx, \
        self.all_select_experts, \
        self.all_select_weights, \
        self.dispatch_tokens, \
        self.used_src_tokens, \
        self.expert_dispatch_tokens, \
        self.expert_dispatch_weights, \
        self.scatter_token_id, \
        self.scatter_token_weight, \
        self.expert_dispatch_token_count, \
        self.expert_dispatch_token_offset = get_moe_tokens_info(
            self.num_tokens, self.num_experts, self.topk, 
            ep_size=self.ep_size, ep_rank=self.ep_rank
        )

        # 以下参数决定当前的 moe_gather 的具体数据类型
        self.dtype = self.args_dict.get("dtype", "bfloat16")


    def vendor_parser(self):
        if self.dtype in ["bfloat16"]:
            pass
        else:
            raise ValueError(
                f"MoeGatherOp base impl only support bfloat16 dtype, but got {self.dtype}"
            )


    def vendor_impl(self):
        self.torch_dtype = getattr(torch, self.dtype)

        self.input_tensor_info = {}
        self.output_tensor_info = {}


        self.input_tensor_info["scatter_tokens"] = OpTensorInfo(
            shape=[self.dispatch_tokens, self.hidden_size], 
            dtype=self.torch_dtype, 
            device=self.backend.get_torch_device_name(),
        )

        self.input_tensor_info["scatter_token_id"] = OpTensorInfo(
            shape=[self.dispatch_tokens], 
            dtype=torch.int32, 
            device=self.backend.get_torch_device_name(),
            creator=partial(create_from_list, data=self.scatter_token_id)
        )
        self.input_tensor_info["scatter_token_weight"] = OpTensorInfo(
            shape=[self.dispatch_tokens], 
            dtype=torch.float32, 
            device=self.backend.get_torch_device_name(),
            creator=partial(create_from_list, data=self.scatter_token_weight)
        )

        self.input_tensor_info["residual_tokens"] = OpTensorInfo(
            shape=[self.num_res_tokens_per_rank, self.hidden_size], 
            dtype=self.torch_dtype, 
            device=self.backend.get_torch_device_name(),
        )

        self.output_tensor_info["convergent_tokens"] = OpTensorInfo(
            shape=[self.num_tokens, self.hidden_size], 
            dtype=self.torch_dtype, 
            device=self.backend.get_torch_device_name(),
            creator=torch.zeros
        )


        # calculator
        self.input_tensor_size = sum([
            calc_tensor_size(info) for info in self.input_tensor_info.values()
        ])
        self.output_tensor_size = sum([
            calc_tensor_size(info) for info in self.output_tensor_info.values()
        ])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size


        scatter_tokens_bytes = calc_tensor_size(self.input_tensor_info["scatter_tokens"])

        self.read_bytes = self.input_tensor_size
        self.write_bytes = scatter_tokens_bytes
        self.io_bytes = self.read_bytes + self.write_bytes

        # creator func
        self._create_tensors_func = partial(
            self._create_in_out_tensors, 
            create_inputs=True, 
            create_outputs=True
        )
        self._run_func = self.vendor_impl_run

    def vendor_impl_run(self, tensor_mapping):
        # get pre-allocated input tensors
        scatter_tokens = tensor_mapping["scatter_tokens"]

        scatter_token_id = tensor_mapping["scatter_token_id"]
        scatter_token_weight = tensor_mapping["scatter_token_weight"]

        residual_tokens = tensor_mapping["residual_tokens"]

        # get pre-allocated output tensors
        convergent_tokens = tensor_mapping["convergent_tokens"]
        convergent_tokens[self.res_token_start:self.res_token_end] += residual_tokens * self.res_scale

        # [dispatch_tokens, hidden_size] --> [num_tokens, hidden_size]
        convergent_tokens.index_add_(
            0, scatter_token_id, 
            (scatter_tokens * scatter_token_weight.unsqueeze(-1)).to(self.torch_dtype)
        )

        return convergent_tokens




"""
swiglu ops
"""

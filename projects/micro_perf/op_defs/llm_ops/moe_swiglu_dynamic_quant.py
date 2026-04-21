"""LLM op: moe_swiglu_dynamic_quant (base implementation)."""
from ._common import *

@ProviderRegistry.register_base_impl("moe_swiglu_dynamic_quant", "ComputeEngine")
class MoeSwigluDynamicQuantOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare_args(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise ValueError(
                f"MoeSwigluDynamicQuantOp only support llm arg_type, but got {self.arg_type}"
            )
        
        # pre-defined attrs
        self.num_tokens = self.args_dict["num_tokens"]
        self.hidden_size = self.args_dict["hidden_size"]

        # moe info
        self.num_experts = self.args_dict["num_experts"]
        self.topk = self.args_dict["topk"]

        # parallel info
        self.ep_size = self.args_dict.get("ep_size", 1)
        self.ep_rank = self.args_dict.get("ep_rank", 0)

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


        # 以下参数决定 moe_swiglu_dynamic_quant 的具体数据类型
        self.dtype = self.args_dict.get("dtype", "bfloat16")
        self.dst_dtype = self.args_dict.get("dst_dtype", "int8")
        

    def vendor_parser(self):
        if self.dtype == "bfloat16" and self.dst_dtype == "int8":
            pass
        else:
            raise ValueError(
                f"MoeSwigluDynamicQuantOp base impl not support dtype {self.dtype} dst_dtype {self.dst_dtype}"
            )

    def vendor_impl(self):
        self.torch_dtype = get_torch_dtype(self.dtype)
        self.dst_torch_dtype = get_torch_dtype(self.dst_dtype)
        
        # input/output tensors
        self.input_tensor_info = {
            "scatter_tokens": OpTensorInfo(
                shape=[self.dispatch_tokens, self.hidden_size * 2], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(),
            ), 
            "experts_smooth_scale": OpTensorInfo(
                shape=[self.num_experts_per_rank, self.hidden_size], 
                dtype=torch.float32, 
                device=self.backend.get_torch_device_name(),
                creator=torch.ones
            ), 
            "experts_token_count": OpTensorInfo(
                shape=[self.num_experts_per_rank], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(
                    self.expert_dispatch_token_count, dtype=dtype, device=device)
            ), 
            "experts_token_offset": OpTensorInfo(
                shape=[self.num_experts_per_rank], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(
                    self.expert_dispatch_token_offset, dtype=dtype, device=device)
            )
        }
        self.output_tensor_info = {
            "quant_tokens": OpTensorInfo(
                shape=[self.dispatch_tokens, self.hidden_size],
                dtype=self.dst_torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
            "per_token_scale": OpTensorInfo(
                shape=[self.dispatch_tokens],
                dtype=torch.float32,
                device=self.backend.get_torch_device_name(),
            ),
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
            create_outputs=True
        )
        self._run_func = self.vendor_impl_run

    def vendor_impl_run(self, tensor_mapping): 
        # get pre-allocated input tensors
        scatter_tokens = tensor_mapping["scatter_tokens"]
        experts_smooth_scale = tensor_mapping["experts_smooth_scale"]
        experts_token_count = tensor_mapping["experts_token_count"]
        experts_token_offset = tensor_mapping["experts_token_offset"]

        # get per-allocated output tensors
        quant_tokens = tensor_mapping["quant_tokens"]
        per_token_scale = tensor_mapping["per_token_scale"]

        # swiglu, x1 used as gating, x2 used as up
        x1, x2 = torch.chunk(scatter_tokens, 2, dim=-1)
        swiglu_tokens = torch.mul(torch.nn.functional.silu(x1), x2)

        # per expert dynamic quant
        for expert_idx in range(self.num_experts_per_rank):
            cur_token_start = self.expert_dispatch_token_offset[expert_idx]
            cur_token_end = cur_token_start + self.expert_dispatch_token_count[expert_idx]

            quant_tokens[cur_token_start:cur_token_end], \
            per_token_scale[cur_token_start:cur_token_end] = \
                smooth_per_token_dynamic_quant(
                    swiglu_tokens[cur_token_start:cur_token_end], 
                    experts_smooth_scale[expert_idx], 
                    dst_torch_dtype=self.dst_torch_dtype
                )

        return quant_tokens, per_token_scale


"""LLM op: moe_scatter_dynamic_quant (base implementation)."""
from ._common import *

@ProviderRegistry.register_base_impl("moe_scatter_dynamic_quant", "ComputeEngine")
class MoeScatterDynamicQuantOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare_args(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise ValueError

        self.dtype = self.args_dict["dtype"]
        self.dst_dtype = self.args_dict["dst_dtype"]

        self.num_tokens = self.args_dict["num_tokens"]
        self.hidden_size = self.args_dict["hidden_size"]

        self.num_experts = self.args_dict["num_experts"]
        self.topk = self.args_dict["topk"]

        self.ep_size = self.args_dict.get("ep_size", 1)
        self.ep_rank = self.args_dict.get("ep_rank", 0)

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

    def vendor_parser(self):
        if self.dtype not in ["bfloat16"]:
            raise ValueError
        if self.dst_dtype not in ["int8", "float8"]:
            raise ValueError

    def vendor_impl(self):
        self.torch_dtype = get_torch_dtype(self.dtype)
        self.dst_torch_dtype = get_torch_dtype(self.dst_dtype)

        self.input_tensor_info = {
            "hidden_states": OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(),
            ), 
            "experts_smooth_scale": OpTensorInfo(
                shape=[self.num_experts, self.hidden_size], 
                dtype=torch.float32, 
                device=self.backend.get_torch_device_name(),
                creator=torch.ones
            ), 
            "selected_experts": OpTensorInfo(
                shape=[self.num_tokens, self.topk], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(self.all_select_experts, dtype=dtype, device=device)
            ), 
            # complete moe_weights
            "moe_weights": OpTensorInfo(
                shape=[self.num_tokens, self.topk], 
                dtype=torch.float32, 
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(self.all_select_weights, dtype=dtype, device=device)
            ), 
        }
        self.output_tensor_info = {
            "scatter_tokens": OpTensorInfo(
                shape=[self.dispatch_tokens, self.hidden_size], 
                dtype=self.dst_torch_dtype, 
                device=self.backend.get_torch_device_name(),
                creator=torch.zeros
            ), 
            "scatter_per_token_scale": OpTensorInfo(
                shape=[self.dispatch_tokens], 
                dtype=torch.float32, 
                device=self.backend.get_torch_device_name(), 
                creator=torch.ones
            ), 
            "scatter_token_id": OpTensorInfo(
                shape=[self.dispatch_tokens], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(
                    self.scatter_token_id, dtype=dtype, device=device)
            ), 
            "scatter_token_weight": OpTensorInfo(
                shape=[self.dispatch_tokens], 
                dtype=torch.float32, 
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(
                    self.scatter_token_weight, dtype=dtype, device=device)
            ), 
            "experts_token_count": OpTensorInfo(
                shape=[self.num_experts_per_rank], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(
                    self.expert_dispatch_token_count, dtype=dtype, device=device)
            ), 
            "experts_token_offset": OpTensorInfo(
                shape=[self.num_experts_per_rank + 1], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(
                    self.expert_dispatch_token_offset, dtype=dtype, device=device)
            )
        }

        # calculator
        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = \
            calc_tensor_size(self.input_tensor_info["hidden_states"]) / self.num_tokens * self.used_src_tokens + \
            calc_tensor_size(self.input_tensor_info["experts_smooth_scale"]) + \
            calc_tensor_size(self.input_tensor_info["selected_experts"]) + \
            calc_tensor_size(self.input_tensor_info["moe_weights"])
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
        hidden_states = tensor_mapping["hidden_states"]
        experts_smooth_scale = tensor_mapping["experts_smooth_scale"]
        selected_experts = tensor_mapping["selected_experts"]
        moe_weights = tensor_mapping["moe_weights"]
        
        # get pre-allocated output tensors
        scatter_tokens = tensor_mapping["scatter_tokens"]
        scatter_per_token_scale = tensor_mapping["scatter_per_token_scale"]

        # For ease of reference in code demonstration, 
        # all the following tensors are precomputed. 
        # Vendors are required to implement the corresponding computation logic during integration.
        scatter_token_id = tensor_mapping["scatter_token_id"]
        scatter_token_weight = tensor_mapping["scatter_token_weight"]
        experts_token_count = tensor_mapping["experts_token_count"]
        experts_token_offset = tensor_mapping["experts_token_offset"]

        for expert_idx in range(self.num_experts_per_rank):
            cur_token_start = self.expert_dispatch_token_offset[expert_idx]
            cur_token_end = cur_token_start + self.expert_dispatch_token_count[expert_idx]
            src_token_indices = scatter_token_id[cur_token_start:cur_token_end]

            scatter_tokens[cur_token_start:cur_token_end], \
            scatter_per_token_scale[cur_token_start:cur_token_end] = \
                smooth_per_token_dynamic_quant(
                    hidden_states[src_token_indices], 
                    experts_smooth_scale[expert_idx], 
                    dst_torch_dtype=self.dst_torch_dtype
                )
            
        return scatter_tokens, scatter_per_token_scale, \
               scatter_token_id, scatter_token_weight, \
               experts_token_count, experts_token_offset



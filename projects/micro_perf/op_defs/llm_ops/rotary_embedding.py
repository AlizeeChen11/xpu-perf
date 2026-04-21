"""LLM op: rotary_embedding (base implementation)."""
from ._common import *

@ProviderRegistry.register_base_impl("rotary_embedding", "ComputeEngine")
class RotaryEmbeddingOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare_args(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm", "batch_llm"]:
            raise ValueError
        
        self.attn_mode = self.args_dict.get("attn_mode", "prefill")
        if not self.attn_mode in ["prefill", "decode"]:
            raise ValueError
        get_attn_info(self.arg_type, self.attn_mode, self.args_dict, self)

        # pre-defined attrs
        self.q_head_num = self.args_dict["q_head_num"]
        self.kv_head_num = self.args_dict["kv_head_num"]
        self.total_head_num = self.q_head_num + 2 * self.kv_head_num
        self.head_dim = self.args_dict["head_dim"]
        self.rope_offset = self.args_dict.get("rope_offset", 0)
        self.rope_dim = self.args_dict["rope_dim"]

        # 以下参数决定当前 rotary_embedding 的计算类型
        self.dtype = self.args_dict.get("dtype", "bfloat16")


    def vendor_parser(self):
        if self.dtype != "bfloat16":
            raise ValueError("RotaryEmbeddingOp only support bfloat16 dtype")

    def vendor_impl(self):
        self.torch_dtype = get_torch_dtype(self.dtype)

        cos_tensor, sin_tensor = precompute_freqs_cis(self.max_kv_len, self.rope_dim)


        self.input_tensor_info = {}
        self.output_tensor_info = {}


        """
        输入的 qkv, packed模式, unsplited模式
        """
        self.input_tensor_info["packed_qkv"] = OpTensorInfo(
            shape=[self.num_tokens, self.total_head_num, self.head_dim], 
            dtype=self.torch_dtype, 
            device=self.backend.get_torch_device_name(),
        )

        """
        确定当前的 num_tokens 中的具体的组成信息
        """
        self.attn_info_tensors = {
            "q_lens": OpTensorInfo(
                shape=[self.batch_size], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(), 
                creator=lambda size, dtype, device: torch.tensor(self.q_lens, dtype=dtype, device=device)
            ), 
            "cache_lens": OpTensorInfo(
                shape=[self.batch_size], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(), 
                creator=lambda size, dtype, device: torch.tensor(self.cache_lens, dtype=dtype, device=device)
            ), 
            "kv_lens": OpTensorInfo(
                shape=[self.batch_size], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(), 
                creator=lambda size, dtype, device: torch.tensor(self.kv_lens, dtype=dtype, device=device)
            ), 
            "accum_q_lens": OpTensorInfo(
                shape=[self.batch_size + 1], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(), 
                creator=lambda size, dtype, device: torch.tensor(self.accum_q_lens, dtype=dtype, device=device)
            ), 
            "accum_cache_lens": OpTensorInfo(
                shape=[self.batch_size + 1], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(), 
                creator=lambda size, dtype, device: torch.tensor(self.accum_cache_lens, dtype=dtype, device=device)
            ), 
            "accum_kv_lens": OpTensorInfo(
                shape=[self.batch_size + 1], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(), 
                creator=lambda size, dtype, device: torch.tensor(self.accum_kv_lens, dtype=dtype, device=device)
            ), 
        }
        self.input_tensor_info.update(self.attn_info_tensors)

        self.sin_cos_info = {
            "cos": OpTensorInfo(
                shape=[self.max_kv_len, self.rope_dim], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(), 
                creator=lambda size, dtype, device: cos_tensor.to(dtype=dtype, device=device)
            ), 
            "sin": OpTensorInfo(
                shape=[self.max_kv_len, self.rope_dim], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(), 
                creator=lambda size, dtype, device: sin_tensor.to(dtype=dtype, device=device)
            ), 
        }
        self.input_tensor_info.update(self.sin_cos_info)



        # calculator
        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.read_bytes -= \
            calc_tensor_size(self.input_tensor_info["packed_qkv"]) \
                / self.total_head_num * self.kv_head_num

        self.write_bytes = \
            calc_tensor_size(self.input_tensor_info["packed_qkv"]) \
                / self.total_head_num * (self.q_head_num + self.kv_head_num)
        
        self.io_bytes = self.read_bytes + self.write_bytes


        # creator func
        self._create_tensors_func = partial(
            self._create_in_out_tensors, 
            create_inputs=True, 
            create_outputs=False
        )
        self._run_func = self.vendor_impl_run

    def vendor_impl_run(self, tensor_mapping):
        # get pre-allocated input tensors
        packed_qkv = tensor_mapping["packed_qkv"]
        q_lens = tensor_mapping["q_lens"]
        accum_q_lens = tensor_mapping["accum_q_lens"]
        cache_lens = tensor_mapping["cache_lens"]

        cos = tensor_mapping["cos"]
        sin = tensor_mapping["sin"]
        

        # for each batch
        for batch_idx in range(self.batch_size):
            q_len = self.q_lens[batch_idx]
            q_offset = self.accum_q_lens[batch_idx]
            cache_len = self.cache_lens[batch_idx]

            token_start = q_offset
            token_end = q_offset + q_len

            qk_head_start = 0
            qk_head_end = self.q_head_num + self.kv_head_num

            dim_start = self.rope_offset
            dim_end = self.rope_offset + self.rope_dim

            cache_start = cache_len
            cache_end = cache_len + q_len

            target_qk = packed_qkv[token_start:token_end, qk_head_start:qk_head_end, dim_start:dim_end].contiguous()
            target_cos = cos[cache_start:cache_end]
            target_sin = sin[cache_start:cache_end]

            packed_qkv[token_start:token_end, qk_head_start:qk_head_end, dim_start:dim_end].copy_(
                rotate(target_qk, target_cos, target_sin)
            )

        return packed_qkv



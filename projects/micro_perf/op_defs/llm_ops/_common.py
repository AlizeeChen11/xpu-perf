"""Shared imports for llm_ops submodules."""
import pathlib
from functools import partial

import torch
from xpu_perf.micro_perf.core.utils import OpTensorInfo, calc_tensor_size, get_torch_dtype, get_torch_dtype_size
from xpu_perf.micro_perf.core.utils import create_from_list
from xpu_perf.micro_perf.core.utils import precompute_freqs_cis, rotate, get_attn_info, get_moe_tokens_info
from xpu_perf.micro_perf.core.utils import smooth_per_token_dynamic_quant, static_quant, fake_quant_gemm
from xpu_perf.micro_perf.core.op import BasicOp, ProviderRegistry


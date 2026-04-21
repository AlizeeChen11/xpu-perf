import pathlib
from functools import partial

from xpu_perf.micro_perf.core.op import ProviderRegistry


try:
    from vllm import _custom_ops as vllm_ops

    @ProviderRegistry.register_vendor_impl("rms_norm", "vllm")
    class VLLMRMSNormOp:
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)
            self.extra_providers = ["vllm"]

        def vendor_impl(self):
            super().vendor_impl()
            if self.add_residual:
                self._create_tensors_func = partial(
                    self._create_in_out_tensors,
                    create_inputs=True,
                    create_outputs=False,
                )
            else:
                self._create_tensors_func = partial(
                    self._create_in_out_tensors,
                    create_inputs=True,
                    create_outputs=True,
                )

        def vendor_impl_run(self, tensor_mapping):
            src = tensor_mapping["src"]
            weight = tensor_mapping["weight"]
            orig_shape = src.shape
            src = src.view(-1, src.shape[-1])

            if self.add_residual:
                residual = tensor_mapping["residual"]
                residual = residual.view(-1, residual.shape[-1])
                vllm_ops.fused_add_rms_norm(src, residual, weight, self.epsilon)
                return src.view(orig_shape)
            else:
                dst = tensor_mapping["dst"]
                vllm_ops.rms_norm(dst, src, weight, self.epsilon)
                return dst.view(orig_shape)

except:
    pass

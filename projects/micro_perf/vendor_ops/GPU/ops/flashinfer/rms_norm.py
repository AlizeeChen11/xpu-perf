import pathlib
from functools import partial

from xpu_perf.micro_perf.core.op import ProviderRegistry


try: 
    from flashinfer.norm import fused_add_rmsnorm, rmsnorm

    @ProviderRegistry.register_vendor_impl("rms_norm", "flashinfer")
    class FlashInferRMSNormOp:
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)
            self.extra_providers = ["flashinfer"]

        def vendor_impl_run(self, tensor_mapping):
            src = tensor_mapping["src"]
            weight = tensor_mapping["weight"]
            orig_shape = src.shape
            src = src.view(-1, src.shape[-1])

            if self.add_residual:
                residual = tensor_mapping["residual"]
                residual = residual.view(-1, residual.shape[-1])
                fused_add_rmsnorm(src, residual, weight, self.epsilon)
                return src.view(orig_shape)
            else:
                dst = rmsnorm(src, weight, self.epsilon)
                return dst.view(orig_shape)

except:
    pass

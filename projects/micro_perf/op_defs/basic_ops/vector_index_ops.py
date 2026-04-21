import pathlib
import torch
from functools import partial
import random

from xpu_perf.micro_perf.core.utils import OpTensorInfo, calc_tensor_size
from xpu_perf.micro_perf.core.op import BasicOp, ProviderRegistry




@ProviderRegistry.register_base_impl("index_select", "ComputeEngine")
class IndexSelectOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare_args(self):
        self.arg_type = self.args_dict["arg_type"]
        self.dtype = self.args_dict["dtype"]
        self.torch_dtype = getattr(torch, self.dtype)

        if self.arg_type == "default":
            self.src_batch_size = self.args_dict["src_batch_size"]
            self.dst_batch_size = self.args_dict["dst_batch_size"]
            self.dim_size = self.args_dict["dim_size"]

            self.input_tensor_info = {
                "src": OpTensorInfo(
                    shape=[self.src_batch_size, self.dim_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                ), 
                "index": OpTensorInfo(
                    shape=[self.dst_batch_size],
                    dtype=torch.int64,
                    device=self.backend.get_torch_device_name(),
                )
            }
            self.output_tensor_info = {
                "dst": OpTensorInfo(
                    shape=[self.dst_batch_size, self.dim_size], 
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                )
            }

            self.src_tensor_size = calc_tensor_size(self.input_tensor_info["src"])
            self.index_tensor_size = calc_tensor_size(self.input_tensor_info["index"])
            self.dst_tensor_size = calc_tensor_size(self.output_tensor_info["dst"])

            self.input_tensor_size = self.src_tensor_size + self.index_tensor_size
            self.output_tensor_size = self.dst_tensor_size
            self.tensor_size = self.input_tensor_size + self.output_tensor_size

            self.read_bytes = self.dst_tensor_size + self.index_tensor_size
            self.write_bytes = self.dst_tensor_size
            self.io_bytes = self.read_bytes + self.write_bytes
        else:
            raise NotImplementedError

    def create_tensors(self, instance_num):
        cur_device = self.backend.get_torch_device_name()

        all_tensor_list = []
        for _ in range(instance_num):
            tensor_mapping = {}

            src = torch.randn(
                size=(self.src_batch_size, self.dim_size), 
                dtype=self.torch_dtype, 
                device=cur_device
            )

            # dst_batch_size indices
            # value vary from [0, src_batch_size)
            random_index = []
            for value in range(self.dst_batch_size):
                random_index.append(value % self.src_batch_size)
            random.shuffle(random_index)
            index = torch.tensor(
                random_index, 
                dtype=torch.int64, 
                device=cur_device
            )

            tensor_mapping["src"] = src
            tensor_mapping["index"] = index
            all_tensor_list.append(tensor_mapping)

        return all_tensor_list


    def vendor_impl_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        index = tensor_mapping["index"]
        dst = torch.index_select(
            input=src, 
            dim=0, 
            index=index
        )
        return dst
    

@ProviderRegistry.register_base_impl("gather", "ComputeEngine")
class GatherOp(IndexSelectOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def create_tensors(self, instance_num):
        cur_device = self.backend.get_torch_device_name()

        all_tensor_list = []
        for _ in range(instance_num):
            tensor_mapping = {}

            src = torch.randn(
                size=(self.src_batch_size, self.dim_size), 
                dtype=self.torch_dtype, 
                device=cur_device
            )

            # dst_batch_size indices
            # value vary from [0, src_batch_size)
            random_index = []
            for value in range(self.dst_batch_size):
                random_index.append(value % self.src_batch_size)
            random.shuffle(random_index)
            index = torch.tensor(
                random_index, 
                dtype=torch.int64, 
                device=cur_device
            ).view(self.dst_batch_size, 1).expand(self.dst_batch_size, self.dim_size)

            tensor_mapping["src"] = src
            tensor_mapping["index"] = index
            all_tensor_list.append(tensor_mapping)

        return all_tensor_list
    
    def vendor_impl_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        index = tensor_mapping["index"]
        dst = torch.gather(
            input=src, 
            dim=0, 
            index=index
        )
        return dst
    


@ProviderRegistry.register_base_impl("embedding", "ComputeEngine")
class EmbeddingOp(IndexSelectOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def vendor_impl_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        index = tensor_mapping["index"]
        dst = torch.torch.nn.functional.embedding(
            input=index, 
            weight=src
        )
        return dst


@ProviderRegistry.register_base_impl("scatter", "ComputeEngine")
class ScatterOp(IndexSelectOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

        self.read_bytes = self.src_tensor_size + self.index_tensor_size
        self.write_bytes = self.src_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

    def create_tensors(self, instance_num):
        cur_device = self.backend.get_torch_device_name()

        all_tensor_list = []
        for _ in range(instance_num):
            tensor_mapping = {}

            src = torch.randn(
                size=(self.src_batch_size, self.dim_size), 
                dtype=self.torch_dtype, 
                device=cur_device
            )

            # dst_batch_size indices
            # value vary from [0, src_batch_size)
            random_index = []
            for value in range(self.src_batch_size):
                random_index.append(value % self.dst_batch_size)
            random.shuffle(random_index)
            index = torch.tensor(
                random_index, 
                dtype=torch.int64, 
                device=cur_device
            ).view(self.dst_batch_size, 1).expand(self.dst_batch_size, self.dim_size)

            dst = torch.randn(
                size=(self.dst_batch_size, self.dim_size), 
                dtype=self.torch_dtype, 
                device=cur_device
            )

            tensor_mapping["src"] = src
            tensor_mapping["index"] = index
            tensor_mapping["dst"] = dst
            all_tensor_list.append(tensor_mapping)

        return all_tensor_list


    def vendor_impl_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        index = tensor_mapping["index"]
        dst = tensor_mapping["dst"]

        dst.scatter_(
            dim=0, 
            index=index, 
            src=src
        )
        return dst


@ProviderRegistry.register_base_impl("index_add", "ComputeEngine")
class IndexAddOp(IndexSelectOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

        self.read_bytes = 2 * self.src_tensor_size + self.index_tensor_size
        self.write_bytes = self.src_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

    def create_tensors(self, instance_num):
        cur_device = self.backend.get_torch_device_name()

        all_tensor_list = []
        for _ in range(instance_num):
            tensor_mapping = {}

            src = torch.randn(
                size=(self.src_batch_size, self.dim_size), 
                dtype=self.torch_dtype, 
                device=cur_device
            )

            # dst_batch_size indices
            # value vary from [0, src_batch_size)
            random_index = []
            for value in range(self.src_batch_size):
                random_index.append(value % self.dst_batch_size)
            random.shuffle(random_index)
            index = torch.tensor(
                random_index, 
                dtype=torch.int64, 
                device=cur_device
            )

            dst = torch.randn(
                size=(self.dst_batch_size, self.dim_size), 
                dtype=self.torch_dtype, 
                device=cur_device
            )

            tensor_mapping["src"] = src
            tensor_mapping["index"] = index
            tensor_mapping["dst"] = dst
            all_tensor_list.append(tensor_mapping)

        return all_tensor_list




    def vendor_impl_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        index = tensor_mapping["index"]
        dst = tensor_mapping["dst"]

        dst.index_add_(
            dim=0, 
            index=index,
            source=src
        )
        return src
    

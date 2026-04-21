# op 设计与实现


## 整体架构

```
ProviderRegistry          注册中心, 管理基类/vendor实现/provider信息
    ├── register_base_impl(op_name, engine_name)       注册算子基类
    ├── register_vendor_impl(op_name, provider_name)   注册vendor实现
    ├── register_provider_info(provider_name, info)    注册provider环境信息
    ├── load_all_base_impls()                          扫描 core/ops/ 加载所有基类
    └── load_all_vendor_impls(backend_name)            扫描 backends/{name}/ops/ 加载所有vendor实现

BasicOp                   算子基类, 定义统一的生命周期
    ├── prepare()          调用链: prepare_args -> vendor_parser -> vendor_impl
    ├── create_tensors()   根据 tensor_info 创建输入输出 tensor
    ├── core_run()         调用 _run_func 执行算子
    └── summary()          根据 latency 计算带宽/算力等指标
```


## ProviderRegistry

管理算子注册的静态类, 维护以下映射表:

| 映射表 | 说明 |
|--------|------|
| `ENGINE_OPS` | engine_name -> {op_name, ...} |
| `OP_ENGINE_MAPPING` | op_name -> engine_name |
| `BASE_IMPL_MAPPING` | op_name -> base_class |
| `OP_MAPPING` | op_name -> {provider_name: impl_class, ...} |
| `OP_PROVIDER_MAPPING` | provider_name -> {op_name: impl_class, ...} |
| `PROVIDER_INFO` | provider_name -> {pkg: version, ...} |

### 注册基类

通过 `@ProviderRegistry.register_base_impl(op_name, engine_name)` 装饰器注册, 定义在 `core/ops/` 下:

```python
@ProviderRegistry.register_base_impl("gemm", "ComputeEngine")
class GemmOp(BasicOp):
    def prepare_args(self): ...
    def vendor_parser(self): ...
    def vendor_impl(self): ...
    def vendor_impl_run(self, tensor_mapping): ...
```

### 注册 vendor 实现

通过 `@ProviderRegistry.register_vendor_impl(op_name, provider_name)` 装饰器注册, 定义在 `backends/{backend}/ops/{provider}/` 下:

```python
@ProviderRegistry.register_vendor_impl("gemm", "torch")
class GPUGemmOp:
    def vendor_parser(self):
        super().vendor_parser()
        # 额外的vendor参数校验
    def vendor_impl(self):
        super().vendor_impl()
        # 额外的vendor初始化
    def vendor_impl_run(self, tensor_mapping):
        # vendor具体的执行逻辑
```

vendor 类无需显式继承基类 -- `register_vendor_impl` 会自动构造 `(vendor_cls, base_cls)` 的派生类, 使 vendor 中未定义的方法回落到基类。

### 注册 provider 信息

在 `backends/{backend}/ops/{provider}/__init__.py` 中检测库是否可用并注册版本信息:

```python
import importlib.metadata
from xpu_perf.micro_perf.core.op import ProviderRegistry

try:
    import flash_attn
    ProviderRegistry.register_provider_info("flash_attn_v2", {
        "flash_attn": importlib.metadata.version("flash_attn")
    })
except Exception:
    pass
```

### 加载流程

`load_all_vendor_impls(backend_name)` 执行时:
1. 先调用 `load_all_base_impls()` 扫描 `core/ops/**/*.py` 加载所有基类
2. 清空 `OP_MAPPING` / `OP_PROVIDER_MAPPING` / `PROVIDER_INFO`
3. 扫描 `backends/{backend_name}/ops/**/*.py`, 逐个 import (跳过 `_` 开头和 `tests/` 目录)
4. Python 导入子目录模块时自动先执行该目录的 `__init__.py`, 完成 provider 信息注册
5. 各 `@register_vendor_impl` 装饰器在 import 时触发, 完成 vendor 实现注册
6. 未被任何 vendor 覆盖的 op 自动回落到 base 实现


## BasicOp

所有算子的基类, 定义统一的初始化和执行生命周期。

### 初始化流程

`__init__` 中按以下顺序初始化:
1. 保存 `args_dict`, `backend`, 分布式信息 (`op_group`, `group_size`)
2. 设置默认的 `_create_tensors_func` 和 `_run_func`
3. 初始化 tensor 信息、size、bytes、flops 等指标为零值
4. 调用 `prepare()`

### prepare 调用链

```python
def prepare(self):
    self.prepare_args()     # 1. 解析参数
    self.vendor_parser()    # 2. 校验参数
    self.vendor_impl()      # 3. 初始化实现
```

每个算子基类需要实现以下方法:

| 方法 | 职责 |
|------|------|
| `prepare_args()` | 从 `self.args_dict` 提取参数, 计算派生属性 (如 flops) |
| `vendor_parser()` | 校验参数组合是否被当前实现支持, 不支持则 raise |
| `vendor_impl()` | 设置 `input_tensor_info` / `output_tensor_info`, 计算 size/bytes, 设置 `_create_tensors_func`, 最后设置 `self._run_func = self.vendor_impl_run` |
| `vendor_impl_run(tensor_mapping)` | 具体的执行逻辑, 接收 tensor 字典, 返回结果 |

### 默认行为

| 属性/方法 | 默认值 |
|-----------|--------|
| `BasicOp.prepare_args()` | `pass` |
| `BasicOp.vendor_parser()` | `pass` |
| `BasicOp.vendor_impl()` | `self._run_func = self.vendor_impl_run` |
| `BasicOp.vendor_impl_run()` | `raise NotImplementedError` |
| `_create_tensors_func` | `partial(_create_in_out_tensors, create_inputs=True, create_outputs=True)` |

### vendor 实现的重写规则

vendor 实现可以选择性重写以上方法:
- 重写 `vendor_parser()` 时应先调用 `super().vendor_parser()` 保留基类校验
- 重写 `vendor_impl()` 时应先调用 `super().vendor_impl()` 保留基类的 tensor 设置和 `_run_func` 赋值, 然后追加 vendor 特有的逻辑 (如修改 `_create_tensors_func` 或覆盖 `_run_func`)
- 重写 `vendor_impl_run()` 提供 vendor 特有的执行逻辑

### 计算指标

| 属性 | 含义 |
|------|------|
| `input_tensor_size` / `output_tensor_size` / `tensor_size` | 内存分配大小 |
| `read_bytes` / `write_bytes` / `io_bytes` | 实际读写字节数 (用于计算带宽) |
| `algo_size` / `bus_size` | 通信算法量 / 总线量 (用于集合通信算子) |
| `calc_flops` | 浮点运算量 |

### summary

`summary(latency_us)` 根据 `is_concurrent` 区分两种报告:
- 计算算子: 输出 `mem_bw(GB/s)`, `calc_flops_power(tflops)`, `calc_mem_ratio`
- 通信算子: 输出 `algo_bw(GB/s)`, `bus_bw(GB/s)`


## 目录结构

```
micro_perf/
├── core/
│   ├── op.py                    ProviderRegistry + BasicOp
│   ├── ops/
│   │   ├── tensor_gemm_ops.py   GemmOp
│   │   ├── vector_*.py          AddOp, CosOp, RMSNormOp, ...
│   │   ├── xccl_ops.py          AllReduceOp, AllGatherOp, ...
│   │   └── llm_ops/
│   │       ├── _common.py       共享 imports
│   │       ├── flash_attention.py
│   │       ├── rotary_embedding.py
│   │       └── ...
│   └── ...
└── backends/
    └── {BACKEND}/
        └── ops/
            ├── {provider_name}/
            │   ├── __init__.py       provider 检测 + register_provider_info
            │   ├── gemm.py           @register_vendor_impl("gemm", "provider_name")
            │   └── ...
            └── ...
```

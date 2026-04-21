# trace_gen v2.1

这份文档面向 `trace_gen` 对外发布包的使用者。

默认对外 bundle 名称为 `trace_gen_release_v2.1`。

## 安装

解压 tar.gz 后进入 bundle 目录，执行：

```bash
pip install wheels/trace_gen-2.1-cp39-abi3-*.whl
```

每个 bundle 只包含一个与当前目标架构匹配的 wheel（`x86_64` 或 `aarch64`）。
这个 wheel 是 `cp39-abi3-manylinux`，目标是在该架构上支持 CPython 3.9+。

可以先运行 `trace-gen --help` 查看总入口，再运行 `trace-gen <子命令> --help` 查看某个子命令的完整参数。

## CSV 列定义

生成的 CSV 总是带表头。

- `timestamp`：时间戳，单位为秒
- `q_len`：输入 query 长度，单位为 token
- `kv_len`：可见上下文长度，单位为 token
- `out_len`：输出长度，单位为 token

其中：

- `P`（Prefill）输出列为 `timestamp,q_len,kv_len`
- `D`（Decode）输出列为 `timestamp,kv_len,out_len`

## Phase 说明

- `P` 代表 `Prefill`
- `D` 代表 `Decode`
- 大部分场景同时提供 `P` 和 `D`
- 少数场景天然就是 `Prefill-only`

## 场景对应表

请选择下面的匿名场景表中的 `scene_id` 和 `phase`：

| scene_id | phases | 说明 |
| --- | --- | --- |
| 1 | P&D | 同时提供 Prefill 和 Decode |
| 2 | P&D | 同时提供 Prefill 和 Decode |
| 3 | P&D | 同时提供 Prefill 和 Decode |
| 4 | P&D | 同时提供 Prefill 和 Decode |
| 5 | P&D | 同时提供 Prefill 和 Decode |
| 6 | P | 只有 Prefill |
| 7 | P&D | 同时提供 Prefill 和 Decode |
| 8 | P&D | 同时提供 Prefill 和 Decode |
| 9 | P | 只有 Prefill |
| 10 | P&D | 同时提供 Prefill 和 Decode |
| 11 | P | 只有 Prefill |
| 12 | P&D | 同时提供 Prefill 和 Decode |
| 13 | P | 只有 Prefill |
| 14 | P&D | 同时提供 Prefill 和 Decode |
| 15 | P&D | 同时提供 Prefill 和 Decode |
| 16 | P&D | 同时提供 Prefill 和 Decode |

## 快速开始

先查看 CLI 帮助：

```bash
trace-gen --help
trace-gen generate-hpp --help
```

## 生成方式与 API

对外 `trace_gen` 一共公开四种方式：

1. 静态 `HPP`
2. 静态 `NHPP`
3. 静态 `NB`
4. 动态内存采样

前三种静态方式都做两件事：

1. 从指定 `scene_id` 和 `phase` 采样 `(x, y)` 请求对
2. 用一种时间模型生成时间戳，再写成 CSV

对外公开的 Python API 一共只有 4 个：

- `trace_gen.generate_hpp_csv`
- `trace_gen.generate_nhpp_csv`
- `trace_gen.generate_nb_csv`
- `trace_gen.sample_requests`

`trace-gen` 命令行只是这 4 个公开 API 的薄封装。

### 1. HPP

定义：

- `HPP` 表示 homogeneous Poisson process
- 它使用单一恒定平均速率
- 直观上可以理解为：长期平均到达率固定为 `qps`

CLI：

```bash
trace-gen generate-hpp \
  --scene-id 1 \
  --phase P \
  --n 100000 \
  --seed 7 \
  --qps 50 \
  --csv-out /tmp/scene_001_p_hpp.csv
```

Python API：

```python
from trace_gen import generate_hpp_csv

generate_hpp_csv(
    scene_id=1,
    phase="P",
    n=100000,
    qps=50.0,
    csv_out="/tmp/scene_001_p_hpp.csv",
    seed=7,
)
```

参数说明：

- `scene_id`：匿名场景编号
- `phase`：`P` 或 `D`
- `n`：要生成的行数
- `qps`：平均每秒请求数
- `csv_out`：输出 CSV 路径
- `seed`：随机种子，用于复现
- `pack_path`：Python API 中可选的 pack 覆盖路径

### 2. NHPP

定义：

- `NHPP` 表示 non-homogeneous Poisson process
- 它允许速率随时间变化
- `trace_gen` 采用分段常数速率函数
- `rates` 中的每个值会作用在一个长度为 `bucket_sec` 的时间桶上

CLI：

```bash
trace-gen generate-nhpp \
  --scene-id 1 \
  --phase P \
  --n 100000 \
  --seed 7 \
  --rates 5,20,40,10 \
  --bucket-sec 1.0 \
  --csv-out /tmp/scene_001_p_nhpp.csv
```

Python API：

```python
from trace_gen import generate_nhpp_csv

generate_nhpp_csv(
    scene_id=1,
    phase="P",
    n=100000,
    rates=[5.0, 20.0, 40.0, 10.0],
    bucket_sec=1.0,
    csv_out="/tmp/scene_001_p_nhpp.csv",
    seed=7,
)
```

参数说明：

- `scene_id`：匿名场景编号
- `phase`：`P` 或 `D`
- `n`：要生成的行数
- `rates`：每个时间桶对应的每秒速率
- `bucket_sec`：每个时间桶的时长，单位秒
- `csv_out`：输出 CSV 路径
- `seed`：随机种子，用于复现
- `pack_path`：Python API 中可选的 pack 覆盖路径

### 3. NB

定义：

- `NB` 表示 negative-binomial bucket model
- 它保持与 `qps` 一致的平均速率目标
- 相比 HPP，它允许更强的桶级突发性
- `alpha` 是这个模型的过离散参数

CLI：

```bash
trace-gen generate-nb \
  --scene-id 1 \
  --phase D \
  --n 100000 \
  --seed 7 \
  --qps 20 \
  --alpha 1.2 \
  --bucket-sec 1.0 \
  --csv-out /tmp/scene_001_d_nb.csv
```

Python API：

```python
from trace_gen import generate_nb_csv

generate_nb_csv(
    scene_id=1,
    phase="D",
    n=100000,
    qps=20.0,
    alpha=1.2,
    bucket_sec=1.0,
    csv_out="/tmp/scene_001_d_nb.csv",
    seed=7,
)
```

参数说明：

- `scene_id`：匿名场景编号
- `phase`：`P` 或 `D`
- `n`：要生成的行数
- `qps`：平均每秒请求数
- `alpha`：负二项模型的过离散参数
- `bucket_sec`：时间桶时长，单位秒
- `csv_out`：输出 CSV 路径
- `seed`：随机种子，用于复现
- `pack_path`：Python API 中可选的 pack 覆盖路径

### 4. 动态采样

定义：

- 动态方式只在内存中采样请求
- 它不会自己写 CSV
- 如果你想把 `trace_gen` 集成到 Python 应用或服务循环里，这是首选方式

Python API：

```python
from trace_gen import sample_requests

rows = sample_requests(scene_id=1, phase="P", n=8, seed=7)
for x, y in rows:
    print(x, y)
```

返回值：

- 对于 `P`：每个二元组是 `(q_len, kv_len)`
- 对于 `D`：每个二元组是 `(kv_len, out_len)`

参数说明：

- `scene_id`：匿名场景编号
- `phase`：`P` 或 `D`
- `n`：采样行数
- `seed`：随机种子，用于复现
- `pack_path`：可选的 pack 覆盖路径

## 常见错误

- `invalid_argument:*`：参数本身不合法
- `invalid_selection:scene-or-phase`：所选 `scene_id` 或 `phase` 不在 pack 中
- `missing_input:pack`：指定的 pack 路径不存在
- `runtime_error:decoder`：decoder 运行或输出写入失败

更详细的参数、phase、场景表和 CSV 定义请直接参考本 `README.md`。

## Citation

如果你在论文、公开报告或公开稿件中使用这个软件，请引用：

```bibtex
@inproceedings{cai2026characterizing,
  title={Characterizing Cloud-Native LLM Inference at Bytedance and Exposing Optimization Challenges and Opportunities for Future AI Accelerators},
  author={Cai, Jingwei and Kong, Dehao and Huang, Hantao and Jiang, Zishan and Ma, Zixuan and Guo, Qingyu and Zhang, Zhenxing and Shi, Guiming and Gao, Mingyu and Ma, Kaisheng and others},
  booktitle={2026 IEEE International Symposium on High Performance Computer Architecture (HPCA)},
  pages={1--19},
  year={2026},
  organization={IEEE}
}
```

## FAQ

### 这是开源软件吗？
不是。这是一个 binary-only 的对外发布包。外部 bundle 不提供源码。

### 可以用于学术研究吗？
可以。只要遵守许可协议，就可以用于非商业学术研究。

### 可以在公司或工业研究团队内部使用吗？
可以，但仅限内部非商业研究用途。不能用于生产、对客交付、托管服务或产品部署。

### 可以用于产品或服务吗？
不允许直接用于产品、对客服务、托管服务或生产环境。关于允许用途和限制范围，请以 `LICENSE` 中的正式定义为准。

### 对外使用者应该依赖什么？
只依赖以下稳定契约：

- wheel
- `scene_id`
- `phase`
- 文档中说明的 CLI / Python API

### 可以重新分发这个 wheel 吗？
只能以原始、完整、未修改的形式重新分发，并且必须同时附带许可与所需通知，且不能附加与原许可冲突的条款。

### ByteDance 是否承诺提供支持、更新或维护？
不承诺。除非另有明确说明，软件按现状提供。

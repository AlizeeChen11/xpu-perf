# trace_gen v2.1

This document is for external users of the public `trace_gen` release bundle.

The default bundle name is `trace_gen_release_v2.1`.

## Install

Unpack the tarball, enter the bundle directory, then install the wheel:

```bash
pip install wheels/trace_gen-2.1-cp39-abi3-*.whl
```

Each bundle contains exactly one wheel for the current target architecture (`x86_64` or `aarch64`).
The wheel is `cp39-abi3-manylinux`, so the same wheel is intended to work on CPython 3.9+ on that architecture.

Use `trace-gen --help` to inspect the top-level CLI, and `trace-gen <subcommand> --help` to inspect subcommand-specific options.

## CSV Format

Generated CSV files always include a header row.

- `timestamp`: event timestamp in seconds
- `q_len`: input query length in tokens
- `kv_len`: visible context length in tokens
- `out_len`: generated output length in tokens

For `P` (Prefill), rows are written as `timestamp,q_len,kv_len`.

For `D` (Decode), rows are written as `timestamp,kv_len,out_len`.

## Phase Meanings

- `P` means `Prefill`
- `D` means `Decode`
- Most scenes are split into both `P` and `D`
- A few scenes are inherently `Prefill-only`

## Scene Table

Use the following anonymous scene table when selecting `--scene-id` and `--phase`:

| scene_id | phases | note |
| --- | --- | --- |
| 1 | P&D | Prefill and Decode |
| 2 | P&D | Prefill and Decode |
| 3 | P&D | Prefill and Decode |
| 4 | P&D | Prefill and Decode |
| 5 | P&D | Prefill and Decode |
| 6 | P | Prefill only |
| 7 | P&D | Prefill and Decode |
| 8 | P&D | Prefill and Decode |
| 9 | P | Prefill only |
| 10 | P&D | Prefill and Decode |
| 11 | P | Prefill only |
| 12 | P&D | Prefill and Decode |
| 13 | P | Prefill only |
| 14 | P&D | Prefill and Decode |
| 15 | P&D | Prefill and Decode |
| 16 | P&D | Prefill and Decode |

## Quick Start

Inspect available CLI options:

```bash
trace-gen --help
trace-gen generate-hpp --help
```

## Generation Modes And API

The public `trace_gen` interface exposes four generation styles:

1. static `HPP`
2. static `NHPP`
3. static `NB`
4. dynamic in-memory sampling

The three static modes all do the same two steps:

1. sample `(x, y)` request pairs from the selected `scene_id` and `phase`
2. generate timestamps using one of the three time models

The public Python API consists of exactly four functions:

- `trace_gen.generate_hpp_csv`
- `trace_gen.generate_nhpp_csv`
- `trace_gen.generate_nb_csv`
- `trace_gen.sample_requests`

The `trace-gen` CLI is a thin wrapper around the same four public interfaces.

### 1. HPP

Definition:

- `HPP` means homogeneous Poisson process
- it uses a single constant average rate over time
- in practical terms: requests arrive with a stable long-run average `qps`

CLI:

```bash
trace-gen generate-hpp \
  --scene-id 1 \
  --phase P \
  --n 100000 \
  --seed 7 \
  --qps 50 \
  --csv-out /tmp/scene_001_p_hpp.csv
```

Python API:

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

Parameters:

- `scene_id`: anonymous scene id from the scene table
- `phase`: `P` or `D`
- `n`: number of rows to generate
- `qps`: average requests per second
- `csv_out`: output CSV path
- `seed`: random seed for reproducibility
- `pack_path`: optional override pack path in Python API

### 2. NHPP

Definition:

- `NHPP` means non-homogeneous Poisson process
- the rate is allowed to change over time
- `trace_gen` uses a piecewise-constant rate function
- each value in `rates` is applied for one bucket of length `bucket_sec`

CLI:

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

Python API:

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

Parameters:

- `scene_id`: anonymous scene id from the scene table
- `phase`: `P` or `D`
- `n`: number of rows to generate
- `rates`: per-bucket rates in requests per second
- `bucket_sec`: bucket duration in seconds
- `csv_out`: output CSV path
- `seed`: random seed for reproducibility
- `pack_path`: optional override pack path in Python API

### 3. NB

Definition:

- `NB` means a negative-binomial bucket model
- it keeps the same average rate target as `qps`
- compared with HPP, it allows burstier bucket-level counts
- `alpha` is the overdispersion parameter of this model

CLI:

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

Python API:

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

Parameters:

- `scene_id`: anonymous scene id from the scene table
- `phase`: `P` or `D`
- `n`: number of rows to generate
- `qps`: average requests per second
- `alpha`: negative-binomial overdispersion parameter
- `bucket_sec`: bucket duration in seconds
- `csv_out`: output CSV path
- `seed`: random seed for reproducibility
- `pack_path`: optional override pack path in Python API

### 4. Dynamic Sampling

Definition:

- dynamic mode samples requests in memory
- it does not write a CSV by itself
- this is the preferred mode if you want to integrate `trace_gen` inside a Python application or service loop

Python API:

```python
from trace_gen import sample_requests

rows = sample_requests(scene_id=1, phase="P", n=8, seed=7)
for x, y in rows:
    print(x, y)
```

Return value:

- for `P`: each tuple is `(q_len, kv_len)`
- for `D`: each tuple is `(kv_len, out_len)`

Parameters:

- `scene_id`: anonymous scene id from the scene table
- `phase`: `P` or `D`
- `n`: number of sampled rows
- `seed`: random seed for reproducibility
- `pack_path`: optional override pack path

## Common Errors

- `invalid_argument:*`: one or more CLI arguments are invalid
- `invalid_selection:scene-or-phase`: the selected `scene_id` or `phase` is not present in the pack
- `missing_input:pack`: the override pack path does not exist
- `runtime_error:decoder`: the decoder failed while sampling or writing output

See this `README.md` for valid phases, scene ids, CSV columns, and usage examples.

## Citation

If your use results in a paper, presentation, or public manuscript, please cite:

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

### Is this software open source?
No. This is a binary-only public release. Source code is not provided in the external bundle.

### Can I use it for academic research?
Yes. Academic and scholarly use is permitted for non-commercial research purposes, subject to the license.

### Can I use it inside a company or industrial research lab?
Yes, but only for internal non-commercial research purposes. Production use, customer delivery, hosted services, and product deployment are not permitted.

### Can I use it in a product or service?
Direct use in a product, customer-facing service, hosted service, or production environment is not permitted. Please refer to `LICENSE` for the exact scope and definitions of permitted non-commercial research use.

### What should I depend on as an external user?
Depend only on:

- the wheel
- `scene_id`
- `phase`
- the documented CLI / Python API

### Can I redistribute the wheel?
Only in the original, complete, and unmodified form, and only if you include the license and required notices and do not add inconsistent terms.

### Does ByteDance provide support, updates, or maintenance?
Not necessarily. Unless explicitly stated otherwise, the software is provided as-is.

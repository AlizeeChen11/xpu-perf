import json
import pathlib
import argparse
from functools import partial

from xpu_perf.model_perf.utils import BenchTestCase
from xpu_perf.model_perf.utils import valid_file, valid_dir, load_dir_as_module

from engine import XpuPerfSimEngine

FILE_DIR = pathlib.Path(__file__).parent.absolute()
DEFAULT_MODEL_DIR = FILE_DIR.joinpath("model_zoo")


def parse_arg():
    """解析 xpu_sim bench 客户端命令行参数。

    连接与模型:
        - --ip / --port: 仿真引擎 HTTP 服务地址
        - --model: 模型配置 JSON（必选）

    run_mode（PD 分离）:
        - prefill: 存在 cache_lens 和 q_lens，每条请求的 cache_len、q_len 可不同
        - decode: 存在 cache_lens 和 q_lens，每条请求 q_len 相同、cache_len 可不同

    block_size（cache 形态）:
        - 0: linear cache，[max_batch_size, kv_head_num, max_kv_len, head_dim]，搭配 slot_mapping [batch_size]
        - >0: paged cache，[total_cache_blocks, kv_head_num, block_size, head_dim]，
          搭配 block_table [max_batch_size, max_num_blocks_per_seq]

    测试输入（与 xpu_gpt 客户端约定一致）:
        - 均匀: batch_size, cache_len, q_len
        - 变长: cache_lens, q_lens，以及 slot_mapping 或 block_table

    用例来源:
        - 命令行单测: 仅支持均匀 batch_size / cache_len / q_len
        - --csv / --generator: 批量测试，支持上述两种输入形态；--generator_args 传给生成脚本

    其它:
        - --disable_print_result: 关闭除引擎初始化与 bench 流式进度以外的控制台输出
        - --no_bench_stream: 关闭 bench NDJSON 流式进度，单次 JSON 响应（兼容旧版服务）
    """
    parser = argparse.ArgumentParser(description="XpuSim bench client")

    # model config
    parser.add_argument("--model_zoo", type=valid_dir, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model_config", type=partial(valid_file, required_suffix=".json"), required=True)

    # bench config
    parser.add_argument("--run_mode", type=str, choices=["prefill", "decode"], default="prefill")
    parser.add_argument("--block_size", type=int, default=512)

    # server config
    parser.add_argument("--ip", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=49371)

    # print config
    parser.add_argument(
        "--disable_print_result",
        action="store_true",
        help="关闭除引擎初始化与 bench 流式进度以外的控制台输出。",
    )
    parser.add_argument(
        "--no_bench_stream",
        action="store_true",
        help="关闭 bench NDJSON 流式进度，使用单次 JSON 响应（兼容旧版 micro_perf 服务）。",
    )

    # bench config
    parser.add_argument("--generator", type=partial(valid_file, required_suffix=".py"))
    parser.add_argument("--generator_args", type=str, default="")
    parser.add_argument("--csv", type=partial(valid_file, required_suffix=".csv"))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--cache_len", type=int, default=0)
    parser.add_argument("--q_len", type=int, default=10240)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arg()

    model_zoo_path = args.model_zoo
    model_config_path = args.model_config
    
    load_dir_as_module(model_zoo_path, "model_zoo")
    model_config_dict = json.loads(model_config_path.read_text(encoding="utf-8"))

    endpoint = XpuPerfSimEngine(
        args.ip, args.port, args.run_mode,
        model_config_dict, 
        disable_print_result=args.disable_print_result,
        bench_stream_progress=not args.no_bench_stream,
    )
    test_cases = BenchTestCase.from_args(args)
    endpoint.bench(test_cases)
    del endpoint


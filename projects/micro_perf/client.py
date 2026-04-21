import os
import csv
import sys
import json
import copy
import pathlib
import requests
import argparse
import jsonlines
import prettytable
from functools import partial
from typing import Any, Dict, List


FILE_DIR = pathlib.Path(__file__).parent.absolute()
BYTE_MLPERF_ROOT = FILE_DIR

from xpu_perf.micro_perf.core.common_utils import logger, setup_logger
from xpu_perf.micro_perf.core.common_utils import parse_tasks, parse_workload, export_reports




def get_info_template(url) -> Dict:
    try:
        response = requests.get(url)
        return response.json()
    except Exception as e:
        return {}

def normal_bench_template(input_dict: Dict, url) -> Dict:
    try:
        response = requests.post(url, json={
            "type": "normal", 
            "data": input_dict
        })
        return response.json()
    except Exception as e:
        return {}








def print_server_info(info_dict):
    # common info
    common_pt = prettytable.PrettyTable()
    common_pt.field_names = ["attr", "value"]
    common_pt.align = "l"
    for attr, value in info_dict["common"].items():
        if attr == "numa_configs":
            continue
        else:
            common_pt.add_row([attr, value])    
    print(common_pt)

    # provider info
    provider_pt = prettytable.PrettyTable()
    provider_pt.field_names = ["provider", "version"]
    provider_pt.align = "l"
    for provider, version in info_dict["provider"].items():
        provider_pt.add_row([provider, version])
    print(provider_pt)

    # backend info
    backend_pt = prettytable.PrettyTable()
    backend_pt.field_names = ["attr", "value"]
    backend_pt.align = "l"
    for attr, value in info_dict["backend"].items():
        backend_pt.add_row([attr, value])
    print(backend_pt)

    # runtime info
    runtime_pt = prettytable.PrettyTable()
    runtime_pt.field_names = ["attr", "value"]
    runtime_pt.align = "l"
    for attr, value in info_dict["runtime"].items():
        runtime_pt.add_row([attr, value])
    print(runtime_pt)

    print("\n\n\n")


def parse_args():
    setup_logger("INFO")
    parser = argparse.ArgumentParser()

    # task input
    parser.add_argument("--task_dir", type=str, 
        default=str(BYTE_MLPERF_ROOT.joinpath("workloads", "basic")))
    parser.add_argument("--task", type=str, default="all")
    parser.add_argument("--workload", type=str)

    # report output
    parser.add_argument("--report_dir", type=str, 
        default=str(BYTE_MLPERF_ROOT.joinpath("reports")))

    parser.add_argument("--server_ip", type=str, default="localhost")
    parser.add_argument("--server_port", type=int, default=49371)
    
    args = parser.parse_args()

    return args





if __name__ == '__main__':
    args = parse_args()

    # client apis
    info_url = f"http://{args.server_ip}:{args.server_port}/info"
    bench_url = f"http://{args.server_ip}:{args.server_port}/bench"

    get_info_func = partial(get_info_template, url=info_url)
    normal_bench_func = partial(normal_bench_template, url=bench_url)


    info_dict = get_info_func()
    if not info_dict or "status" in info_dict:
        logger.error("get server info failed.")
        sys.exit(-1)
    print_server_info(info_dict)




    test_cases = {}
    if args.workload is not None:
        test_cases = parse_workload(args.workload)
    else:
        test_cases = parse_tasks(args.task_dir, args.task)
    if not test_cases:
        logger.error("No valid test cases found. Exiting.")
        sys.exit(1)
    print("*" * 100)
    logger.info(f"test cases: ")
    for op_name, op_cases in test_cases.items():
        logger.info(f"{op_name} has {len(op_cases)} test cases")
    print("*" * 100)


    bench_results = normal_bench_func(test_cases)
    if not bench_results or "status" in info_dict:
        logger.error("get no results.")
        sys.exit(-1)

    export_reports(
        args.report_dir, 
        info_dict, 
        test_cases, 
        bench_results
    )


        

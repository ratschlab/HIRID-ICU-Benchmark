''' Script generating the endpoints'''

import argparse
import logging
import os.path
import random
import sys
import ipdb

import gin

from icu_benchmarks.endpoints.endpoint_benchmark import endpoint_gen_benchmark


def execute(configs):
    ''' Dispatch to the correct endpoint generation function'''
    random.seed(configs["random_state"])
    if configs["endpoint"] == "benchmark":
        endpoint_gen_benchmark(configs)
    else:
        logging.info("Invalid endpoint version requested...")


@gin.configurable
def parse_gin_args_endpoints(old_configs, gin_configs=None):
    return {**old_configs, **gin_configs}  # merging dicts


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--debug_mode", default=None, action="store_true", help="Only one batch + do not write to FS")
    parser.add_argument("--verbose", action="store_true", default=None, help="Verbose messages")
    parser.add_argument("--small_sample", action="store_true", default=None, help="Use a small sample of PIDs")
    parser.add_argument("--batch_idx", type=int, default=None, help="Batch index to process")
    parser.add_argument("--split", default=None, help="Split to process")
    parser.add_argument("--run_mode", default=None, help="Execution mode")
    parser.add_argument("--endpoint", default=None, help="Endpoint to process")
    parser.add_argument("--random_state", default=None, help="Random seed to use for endpoint replicates")
    parser.add_argument("--load_batch_at_once", default=True, help="Load batch in one file")

    parser.add_argument("--gin_config", default="./gin_configs/ep_benchmark.gin", help="GIN config to use")

    configs = vars(parser.parse_args())
    gin.parse_config_file(configs["gin_config"])
    configs = parse_gin_args_endpoints(configs)

    configs["VAR_ID"] = configs["VAR_IDS"]
    split_key = configs["split"]
    batch_idx = configs["batch_idx"]
    rseed = configs["random_state"]

    if configs["run_mode"] == "CLUSTER":

        if configs["endpoint"] == "benchmark":
            sys.stdout = open(
                os.path.join(configs["log_dir"], "BENCHMARK_ENDPOINT_GEN_{}_{}.stdout".format(split_key, batch_idx)),
                'w')
            sys.stderr = open(
                os.path.join(configs["log_dir"], "BENCHMARK_ENDPOINT_GEN_{}_{}.stderr".format(split_key, batch_idx)),
                'w')

    execute(configs)

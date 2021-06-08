import argparse
import os.path
import sys

import gin

from icu_benchmarks.labels.label_benchmark import label_gen_benchmark


def execute(configs):
    if configs["endpoint"] == "benchmark":
        label_gen_benchmark(configs)
    else:
        assert (False, "Wrong endpoint mode")


@gin.configurable()
def parse_gin_args_labels(old_configs, gin_configs=None):
    return {**old_configs, **gin_configs}  # merging dicts


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument("--split_key", default=None, help="For which split should labels be produced?")
    parser.add_argument("--batch_idx", type=int, default=None, help="On which batch should this process operate?")
    parser.add_argument("--debug_mode", action="store_true", default=None,
                        help="Debug mode for testing, no output created to file-system")
    parser.add_argument("--run_mode", default=None, help="Execution mode")

    # GIN config
    parser.add_argument("--gin_config", default="./gin_configs/label_benchmark.gin",
                        help="Location of GIN config to load, and overwrite the arguments")

    args = parser.parse_args()
    configs = vars(args)
    gin.parse_config_file(configs["gin_config"])
    configs = parse_gin_args_labels(configs)

    split_key = configs["split_key"]
    batch_idx = configs["batch_idx"]

    if configs["run_mode"] == "CLUSTER":
        sys.stdout = open(
            os.path.join(configs["log_dir"], "LABEL_{}_{}_{}.stdout".format(configs["endpoint"], split_key, batch_idx)),
            'w')
        sys.stderr = open(
            os.path.join(configs["log_dir"], "LABEL_{}_{}_{}.stderr".format(configs["endpoint"], split_key, batch_idx)),
            'w')

    execute(configs)

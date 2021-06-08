import functools
import re
from pathlib import Path

import gin

from icu_benchmarks.common import processing, utils
from icu_benchmarks.common.datasets import Dataset
from icu_benchmarks.endpoints import endpoint_benchmark, endpoint_gen


def process_endpoint_chunk(part_path: Path, imputed_path: Path, output_dir: Path, config):
    # modify config
    config['batch_idx'] = int(re.match('part-([0-9]+).parquet', part_path.name).groups()[0])

    config['imputed_path'] = imputed_path
    config['endpoint_path'] = output_dir

    endpoint_benchmark.endpoint_gen_benchmark(config)


def generate_endpoints(merged_path, imputed_path, output_dir, gin_config_path=None, nr_workers=1):
    if not gin_config_path:
        gin_config_path = utils.get_code_base_root() / 'endpoints/gin_configs/ep_benchmark.gin'

    gin.parse_config_file(gin_config_path)
    config = endpoint_gen.parse_gin_args_endpoints({})

    merged_ds = Dataset(merged_path)
    parts = merged_ds.list_parts()

    output_ds = Dataset(output_dir)
    output_ds.prepare()

    processing.exec_parallel_on_parts(functools.partial(process_endpoint_chunk, imputed_path=imputed_path,
                                                        output_dir=output_dir, config=config), parts, nr_workers)
    output_ds.mark_done()

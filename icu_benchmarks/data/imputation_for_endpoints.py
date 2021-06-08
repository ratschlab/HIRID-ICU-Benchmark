import functools
import re
from pathlib import Path

import gin

from icu_benchmarks.common import processing, utils
from icu_benchmarks.common.datasets import Dataset
from icu_benchmarks.imputation import impute_one_batch


def process_chunk(part_path: Path, output_dir: Path, config):
    config['batch_idx'] = int(re.match('part-([0-9]+).parquet', part_path.name).groups()[0])
    config['bern_reduced_merged_path'] = str(part_path.parent)

    config['bern_imputed_reduced_dir'] = output_dir

    impute_one_batch.execute(config)


def impute_for_endpoints(merged_path, extended_static_data_path, output_dir, gin_config_path=None, nr_workers=1):
    input_ds = Dataset(merged_path)
    parts = input_ds.list_parts()

    if not gin_config_path:
        gin_config_path = utils.get_code_base_root() / 'imputation/gin_configs/impute_dynamic_benchmark.gin'

    gin.parse_config_file(gin_config_path)
    config = impute_one_batch.parse_gin_args_impute({})

    config['bern_static_info_path'] = str(extended_static_data_path)

    output_ds = Dataset(output_dir)
    output_ds.prepare()

    processing.exec_parallel_on_parts(functools.partial(process_chunk, output_dir=output_dir, config=config),
                                      parts, nr_workers)

    output_ds.mark_done()

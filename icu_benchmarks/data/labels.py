import functools
import re
from pathlib import Path

import gin

from icu_benchmarks.common import processing, utils
from icu_benchmarks.common.datasets import Dataset
from icu_benchmarks.labels import label_benchmark, label_gen

batch_parquet_pattern = re.compile('batch_([0-9]+).parquet')


def process_chunk(part_path: Path, endpoints_path: Path, imputation_for_endpoints_path: Path, output_dir: Path, config):
    config['batch_idx'] = int(re.match(batch_parquet_pattern, part_path.name).groups()[0])

    config["label_dir"] = output_dir
    config["endpoint_dir"] = endpoints_path
    config["imputed_dir"] = imputation_for_endpoints_path

    label_benchmark.label_gen_benchmark(config)


def generate_labels(endpoints_path: Path, imputation_for_endpoints_path: Path,
                    exended_general_data_path: Path, output_dir: Path, nr_workers=1, gin_config_path=None):
    input_ds = Dataset(endpoints_path, part_re=batch_parquet_pattern)
    parts = input_ds.list_parts()

    if not gin_config_path:
        gin_config_path = utils.get_code_base_root() / 'labels/gin_configs/label_benchmark.gin'

    gin.parse_config_file(gin_config_path)
    config = label_gen.parse_gin_args_labels({})

    config["general_data_table_path"] = exended_general_data_path

    output_ds = Dataset(output_dir)
    output_ds.prepare()

    processing.exec_parallel_on_parts(functools.partial(process_chunk, endpoints_path=endpoints_path,
                                                        imputation_for_endpoints_path=imputation_for_endpoints_path,
                                                        output_dir=output_dir, config=config),
                                      parts, nr_workers)

    output_ds.mark_done()

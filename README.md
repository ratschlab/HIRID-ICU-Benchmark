# ICU Benchmark Project

This project aim is to build a benchmark for ICU related tasks.

## Setup

In the following we assume a Linux installation, however, other platforms may also work

1. Install Conda, see the [official installation instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. clone this repository and change into the directory of the repository
3. `conda env update` (creates an environment `icu-benchmark`)
4. `pip install -e .`

## Download Data

1. Get access to the HiRID 1.1.1 dataset on [physionet](https://physionet.org/content/hirid/1.1.1/). This entails
   1. getting a [credentialed physionet account](https://physionet.org/settings/credentialing/)
   2. [submit a usage request](https://physionet.org/request-access/hirid/1.1.1/) to the data depositor
2. Once access is granted, download the following files
   1. [reference_data.tar.gz](https://physionet.org/content/hirid/1.1.1/reference_data.tar.gz)
   2. [observation_tables_parquet.tar.gz](https://physionet.org/content/hirid/1.1.1/raw_stage/observation_tables_parquet.tar.gz)
   3. [pharma_records_parquet.tar.gz](https://physionet.org/content/hirid/1.1.1/raw_stage/pharma_records_parquet.tar.gz)
3. unpack the files into the same directory using e.g. `cat *.tar.gz | tar zxvf - -i`



## How to Run

### Run Prepocessing

Activate the conda environment using `conda activate icu-benchmark`. Then

```
icu-benchmarks preprocess --hirid-data-root [path to unpacked parquet files as downloaded from phyiosnet] \
                          --work-dir [output directory] \
                          --var-ref-path ./preprocessing/resources/varref.tsv \
                          --split-path ./preprocessing/resources/split.tsv \
                          --nr-workers 8
```

The above command requires about 6GB of RAM per core and in total approximately 30GB of disk space.


### Run Training

To run a custom training you should, activate the conda environment using `conda activate icu-benchmark`. Then
```
icu-benchmarks train -c [path to gin config] \
                     -l [path to logdir] \
                     -t [task name] \
                     -sd [seed number] 
```
Task name should be one of the following : `Mortality_At24Hours, Dynamic_CircFailure_12Hours, Dynamic_RespFailure_12Hours, Dynamic_UrineOutput_2Hours_Reg, Phenotyping_APACHEGroup` or `Remaining_LOS_Reg`.\\
To see an example of `gin-config` file please refer to `./configs/`. You can also check directly the [gin-config documentation](https://github.com/google/gin-config).\\

Tu run the experiments from the paper you we build pre-defined scripts located in `./run_scripts`. For instance, you can run the following command to reproduce GRU baseline on Mortatility task:
```
sh run_script/baselines/Mortality_At24Hours/GRU.sh
```







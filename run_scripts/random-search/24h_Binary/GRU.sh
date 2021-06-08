source activate icu-benchmark

python -m icu_benchmarks.run train \
                             -c configs/hirid/Classification/GRU.gin \
                             -l logs/random_search/24_binary/GRU/run \
                             -t Mortality_At24Hours \
                             --maxlen 288 \
                             -rs True\
                             -lr  3e-4 1e-4 3e-5 1e-5\
                             -sd 1111 2222 3333 \
                             --hidden 32 64 128 256 \
                             --do 0.0 0.1 0.2 0.3 0.4 \
                             --depth 1 2 3

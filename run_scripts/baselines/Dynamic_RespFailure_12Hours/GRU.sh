source activate icu-benchmark
python -m icu_benchmarks.run train \
                             -c configs/hirid/Classification/GRU.gin \
                             -l logs/benchmark_exp/GRU/ \
                             -t Dynamic_RespFailure_12Hours \
                             -o True \
                             -lr 3e-4\
                             --hidden 64 \
                             --do 0.0 \
                             --depth 3 \
                             -sd 1111 2222 3333 4444 5555 6666 7777 8888 9999 0000


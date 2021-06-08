source activate icu-benchmark
python -m icu_benchmarks.run train \
                             -c configs/hirid/Classification/transformer.gin \
                             -l logs/benchmark_exp/transformer/ \
                             -t Dynamic_RespFailure_12Hours\
                             -o True \
                             -lr 1e-4\
                             -bs 8\
                             --hidden 128 \
                             --do 0.3 \
                             --do_att 0.2 \
                             --depth 1 \
                             --heads 1 \
                             -sd 1111 2222 3333 4444 5555 6666 7777 8888 9999 0000


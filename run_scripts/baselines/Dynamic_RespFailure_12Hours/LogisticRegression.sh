source activate icu-benchmark
python -m icu_benchmarks.run train \
                             -c configs/hirid/Classification/LogisticRegression.gin \
                             -l logs/benchmark_exp/LogisticRegression/ \
                             -t Dynamic_RespFailure_12Hours\
                             -o True \
                             --penalty 'l2' \
                             --c_parameter 0.1 \
                             -sd 1111 2222 3333 4444 5555 6666 7777 8888 9999 0000

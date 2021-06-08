source activate icu-benchmark
python -m icu_benchmarks.run train \
                             -c configs/hirid/Classification/TCN.gin \
                             -l logs/benchmark_exp/TCN/ \
                             -t Dynamic_CircFailure_12Hours\
                             -o True \
                             --hidden 32 \
                             -lr 3e-4\
                             --do 0.1 \
                             --kernel 4 \
                             -sd 1111 2222 3333 4444 5555 6666 7777 8888 9999 0000 \


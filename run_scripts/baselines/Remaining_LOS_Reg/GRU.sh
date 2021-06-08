source activate icu-benchmark
python -m icu_benchmarks.run train \
                             -c configs/hirid/Regression/GRU.gin \
                             -l logs/benchmark_exp/GRU/ \
                             -t Remaining_LOS_Reg\
                             -o True \
                             -lr 3e-4\
                             --hidden 128 \
                             --do 0.3 \
                             --depth 2 \
                             -sd 1111 2222 3333 4444 5555 6666 7777 8888 9999 0000


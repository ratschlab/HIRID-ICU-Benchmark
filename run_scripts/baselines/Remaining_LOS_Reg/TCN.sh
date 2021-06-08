source activate icu-benchmark
python -m icu_benchmarks.run train \
                             -c configs/hirid/Regression/TCN.gin \
                             -l logs/benchmark_exp/TCN/ \
                             -t Remaining_LOS_Reg\
                             -o True \
                             --hidden 128 \
                             -lr 3e-4\
                             --do 0.3 \
                             --kernel 32 \
                             -sd 1111 2222 3333 4444 5555 6666 7777 8888 9999 0000 \


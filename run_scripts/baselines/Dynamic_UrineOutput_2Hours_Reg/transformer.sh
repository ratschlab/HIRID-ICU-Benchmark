source activate icu-benchmark
python -m icu_benchmarks.run train \
                             -c configs/hirid/Regression/transformer.gin \
                             -l logs/benchmark_exp/transformer/ \
                             -t Dynamic_UrineOutput_2Hours_Reg\
                             -o True \
                             -lr 3e-4\
                             -bs 8\
                             --hidden 64 \
                             --do 0.1 \
                             --do_att 0.1 \
                             --depth 1 \
                             --heads 1 \
                             -sd 1111 2222 3333 4444 5555 6666 7777 8888 9999 0000


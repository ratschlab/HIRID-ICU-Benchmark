source activate icu-benchmark
python -m icu_benchmarks.run train \
                             -c configs/hirid/Classification/LGBM.gin \
                             -l logs/benchmark_exp/LGBM/ \
                             -t Phenotyping_APACHEGroup \
                             --loss-weight balanced \
                             --num-class 15 \
                             --maxlen 288 \
                             -o True \
                             --depth 3 \
                             --subsample-feat 0.66 \
                             --subsample-data 0.66 \
                             -sd 1111 2222 3333 4444 5555 6666 7777 8888 9999 0000


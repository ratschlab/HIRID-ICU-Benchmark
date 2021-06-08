source activate icu-benchmark

python -m icu_benchmarks.run train \
                             -c configs/hirid/Classification/LGBM.gin \
                             -l logs/random_search/24_binary/LGBM/run \
                             -t Mortality_At24Hours \
                             --maxlen 288 \
                             -rs True\
                             -sd 1111 2222 3333 \
                             --depth 3 4 5 6 7 \
                             --loss-weight balanced None \
                             --subsample-feat 0.33 0.66 1.00 \
                             --subsample-data 0.33 0.66 1.00

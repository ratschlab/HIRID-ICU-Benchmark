source activate icu-benchmark
python -m icu_benchmarks.run evaluate \
                             -c configs/hirid/Classification/LGBM_w_feat.gin \
                             -l files/pretrained_weights/LGBM_w_feat/ \
                             -t Dynamic_CircFailure_12Hours\
                             -o True \
                             --depth 3 \
                             --subsample-feat 0.66 \
                             --subsample-data 0.33 \


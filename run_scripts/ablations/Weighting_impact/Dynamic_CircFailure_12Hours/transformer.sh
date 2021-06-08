source activate icu-benchmark
python -m icu_benchmarks.run train \
                             -c configs/hirid/Classification/transformer.gin \
                             -l logs/Weighting_impact/transformer/ \
                             -t Dynamic_CircFailure_12Hours\
                             -o True --loss-weight balanced \
                             -lr 1e-4\
                             -bs 8\
                             --hidden 64 \
                             --do 0.0 \
                             --do_att 0.3\
                             --depth 1 \
                             --heads 2 \
                             -sd 1111 2222 3333 4444 5555 6666 7777 8888 9999 0000


source activate icu-benchmark
python -m icu_benchmarks.run evaluate \
                             -c configs/hirid/Classification/transformer.gin \
                             -l files/pretrained_weights/transformer/ \
                             -t Dynamic_RespFailure_12Hours\
                             -o True \
                             -lr 1e-4\
                             -bs 8\
                             --hidden 128 \
                             --do 0.3 \
                             --do_att 0.2 \
                             --depth 1 \
                             --heads 1 \



source activate icu-benchmark
python -m icu_benchmarks.run evaluate \
                             -c configs/hirid/Classification/GRU.gin \
                             -l files/pretrained_weights/GRU/ \
                             -t Dynamic_RespFailure_12Hours \
                             -o True \
                             -lr 3e-4\
                             --hidden 64 \
                             --do 0.0 \
                             --depth 3 \



source activate icu-benchmark
python -m icu_benchmarks.run train \
                             -c configs/hirid/Classification/LSTM.gin \
                             -l logs/benchmark_exp/LSTM/ \
                             -t Phenotyping_APACHEGroup --loss-weight balanced \
                             --num-class 15 \
                             --maxlen 288 \
                             -o True \
                             -lr 3e-4\
                             --hidden 256 \
                             --do 0.0 \
                             --depth 1 \
                             -sd 1111 2222 3333 4444 5555 6666 7777 8888 9999 0000


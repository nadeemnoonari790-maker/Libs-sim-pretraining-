#!/bin/bash
# Dataset 2 (Mount Holyoke ChemLIBS, Mars atmosphere) and the cross-instrument test.
# Requires data/chemlibs/chemlibs_mars5mj.npz and data/chemlibs/libs_metadata.xlsx (see tools/fetch_chemlibs.py).
# Cross-instrument test also needs dataset 1 (run_dataset1.sh) for the deep-sea data and results/main_preds.pkl.
set -e
cd "$(dirname "$0")/.."
for s in 1 2 3; do python3 src/synth2.py --n 24000 --seed $s; done
for s in 1 2 3; do python3 src/exp2.py pretrain --tag synth2 --seed $s; python3 src/exp2.py pretrain --tag elperm2 --seed $s; done
python3 src/exp2.py run --seeds 0 1 2 --ks 4 8 16 32 128 512 0 --methods null pls cnn_real cnn_ft cnn_elperm cnn_mix
python3 src/exp2.py run --seeds 0 1 2 --ks 4 32 0 --methods cnn_head
python3 src/xinst.py pretrain; python3 src/xinst.py run
python3 src/sal2.py run; python3 src/sal2.py analyse
python3 src/evaluate2.py | tee results/summary2.txt

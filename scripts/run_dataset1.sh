#!/bin/bash
# Dataset 1 (deep-sea ChemiCam). Reproduces every dataset-1 table/figure input in results/.
# Requires the ChemiCam open database in data/chemicam_open_database/ (or set CHEMICAM_DIR).
# The original study ran these steps in several resumable stages on a 2-core CPU (roughly 1-2 days in total).
set -e
cd "$(dirname "$0")/.."
python3 src/build_real_cache.py                     # 25,457 block spectra of 68 samples -> cache/
python3 src/test_forward_model.py                   # physics unit tests
python3 src/fit_alloys.py --alloys C2600 --control 0 --tag fig2   # forward-model fit to brass C2600 (Fig. 1)
# synthetic pre-training sets (24,000 spectra each)
for s in 1 2 3; do
  python3 src/synth.py --seed $s --tag informed
  python3 src/synth.py --seed $s --tag broad --prior broad
  python3 src/synth.py --seed $s --tag nonoise --noise none
  python3 src/synth.py --seed $s --tag noambient --ambient 0 --nuisance 0
done
python3 src/synth.py --n 96000 --seed 1 --tag n96000
python3 src/make_controls.py                        # element-permuted labels; 1.5k / 6k subsets
python3 src/make_shuf.py                            # sample-shuffled labels
for t in informed broad nonoise noambient elperm shuf; do for s in 1 2 3; do python3 src/pretrain.py --tag $t --seed $s; done; done
for t in n1500 n6000 n96000; do python3 src/pretrain.py --tag $t --seed 1; done
# main label-budget sweep and controls
python3 src/run_exp.py --seeds 0 1 2
python3 src/run_exp.py --pre_tag shuf   --methods cnn_ft --ks 4 8 0 --seeds 0 1 2 --out results/ctrl_shuf.pkl
python3 src/run_exp.py --pre_tag elperm --methods cnn_ft --ks 4 8 0 --seeds 0 1 2 --out results/ctrl_elperm.pkl
python3 src/run_exp.py --methods cnn_head --ks 4 8 0 --seeds 0 1 2 --out results/head_preds.pkl
for t in broad nonoise noambient; do python3 src/run_exp.py --pre_tag $t --methods cnn_ft --ks 4 8 --seeds 0 1 2 --out results/abl_$t.pkl; done
for t in n1500 n6000 n96000; do python3 src/run_exp.py --pre_tag $t --methods cnn_ft --ks 4 8 --seeds 0 --out results/scale_$t.pkl; done
python3 src/realism.py informed_s1 broad_s1 nonoise_s1 noambient_s1   # classifier two-sample test
python3 src/adabn.py                                                   # label-free BN adaptation
LIBS_CROP=1 python3 src/realism.py informed_s1 broad_s1 nonoise_s1     # edge-band robustness
LIBS_CROP=1 python3 src/pretrain.py --tag informed --seed 1
LIBS_CROP=1 python3 src/run_exp.py --seeds 0 --methods null pls cnn_real cnn_ft --ks 4 8 0 --out results/crop_preds.pkl
for p in mlp xmat; do python3 src/exp_new.py --part $p; done
python3 src/exp_new.py --part sal --seeds 0 1
# aggregation
python3 src/evaluate.py; python3 src/extra_eval.py; python3 src/breakdown.py; python3 src/analyze_new.py

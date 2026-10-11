# Simulation-based pre-training for small-sample LIBS quantification

Code, saved predictions and figures for the manuscript

> **Simulation-based pre-training improves small-sample LIBS quantification through correct line–element association**  
> Hamzah (Xidian University), 2026 — *manuscript in preparation*

A homogeneous LTE slab forward model of the laser-induced plasma (Saha–Boltzmann populations, radiative transfer with
self-absorption, per-line Stark widths) feeds a domain-randomised generator of synthetic LIBS spectra. A 1D CNN is
pre-trained on the synthetic spectra and fine-tuned on real labelled spectra from two public databases. Evaluation
holds out whole sites / sample suites and sweeps the number of labelled samples, with negative controls
(sample-shuffled and element-permuted synthetic labels), a frozen-extractor control, generator ablations, attribution
analysis and a cross-instrument test.

## Repository layout

| Path | Content |
|---|---|
| `src/forward_model.py` | LTE slab forward model |
| `src/synth.py`, `src/synth2.py` | Domain-randomised generators for dataset 1 (deep-sea) and dataset 2 (ChemLIBS) |
| `src/preprocess.py` | Preprocessing applied identically to real and synthetic spectra |
| `src/data.py`, `src/data2.py`, `src/build_real_cache.py` | Real-data loading, labels, grouped folds |
| `src/models.py` | CNN, samplers, training loop |
| `src/pretrain.py`, `src/run_exp.py` | Dataset 1: pre-training and label-budget experiments |
| `src/exp2.py` | Dataset 2: pre-training and label-budget experiments |
| `src/xinst.py` | Cross-instrument test (real ChemLIBS vs simulated pre-training for the deep-sea data) |
| `src/exp_new.py`, `src/sal2.py` | Attribution, cross-matrix and MLP analyses |
| `src/make_controls.py`, `src/make_shuf.py`, `src/realism.py`, `src/adabn.py`, `src/fit_alloys.py` | Controls and diagnostics |
| `src/evaluate.py`, `src/extra_eval.py`, `src/breakdown.py`, `src/analyze_new.py`, `src/evaluate2.py` | Aggregation, bootstrap CIs, paired Wilcoxon tests |
| `src/make_figures.py`, `src/fig8.py`, `src/fig9.py`, `src/figstyle.py` | Figures (`src/fig_schematics.py`: schematic Figs. 1–2) |
| `src/test_forward_model.py` | Unit tests of the forward-model physics |
| `scripts/` | `run_dataset1.sh`, `run_dataset2.sh`, `make_figures.sh` — full reproduction in order |
| `tools/fetch_chemlibs.py` | Downloads and reduces the ChemLIBS spectra from NASA PDS |
| `data/` | `atomic_lines.db` and instructions for obtaining the real spectra (see `data/README.md`) |
| `results/` | Saved per-sample predictions (`*.pkl`) and all aggregated numbers (`*.json`) behind the tables |
| `figures/` | All figures (PDF, SVG, 600-dpi PNG); file → manuscript figure: `fig0_overview` → Fig. 1, `fig0_cnn` → Fig. 2 (both also as editable `.drawio`), `fig2_spectra` → Fig. 3, `fig3_label_budget` → Fig. 4, `fig7_mechanism` → Fig. 5, `fig6_ablations` → Fig. 6, `fig4_elements_k8` → Fig. 7, `fig5_parity_k8` → Fig. 8, `fig8_new_analyses` → Fig. 9, `fig9_dataset2` → Fig. 10 |

## Installation

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
CPU is sufficient (the study was run on 2 CPU cores).

## Data

1. **Dataset 1 — ChemiCam deep-sea database** (Takahashi et al. 2020):
   `git clone https://github.com/ocean-perception/chemicam_open_database data/chemicam_open_database`
2. **Dataset 2 — Mount Holyoke ChemLIBS reference spectra** (Dyar & Lepore, NASA PDS, doi:10.17189/b2aj-cz96):
   `python tools/fetch_chemlibs.py` (about 3.5 GB of downloads, only the reduced file is kept).

## Reproducing the results

```bash
bash scripts/run_dataset1.sh     # dataset 1: tables 1-6, figures 3-9
bash scripts/run_dataset2.sh     # dataset 2 and cross-instrument test: tables 7-8, figure 10
bash scripts/make_figures.sh     # figures from results/
```
Synthetic sets, pre-trained weights and data caches are written to `cache/` (several GB; not part of the repository).
All random seeds are fixed. A full rerun takes on the order of two days on a 2-core CPU; individual steps are
resumable. To check the statistics without retraining, run only the aggregation scripts
(`src/evaluate.py`, `src/extra_eval.py`, `src/breakdown.py`, `src/analyze_new.py`, `src/evaluate2.py`) on the saved
predictions in `results/` (dataset caches are still needed for the ground-truth labels).

## Protocol summary

* Targets: log10 of wt% (floor 0.01 wt%); missing labels are masked in losses and metrics.
* Folds: 5 folds; all samples of a site group (dataset 1) or sample suite (dataset 2) are in the same fold.
* Label budgets: 4, 8, 16, 32 (128, 512) and all training samples; three seeds.
* Statistics: seed-mean macro RMSE, group-level bootstrap intervals, paired Wilcoxon tests on group mean absolute errors.
* Nothing was tuned on test folds; dataset 2 used the dataset-1 network and training settings unchanged.

## Citation

Please cite the manuscript (citation to be added on publication) and the two data sources:

* Takahashi T, Yoshino S, Takaya Y, Nozaki T, Ohki K, Ohki T, Sakka T, Thornton B (2020) Deep-Sea Research Part I 158:103232. https://doi.org/10.1016/j.dsr.2020.103232
* Dyar MD, Lepore KH (2026) Multi-Parameter LIBS Reference Database of Geological Materials Bundle, v1.1. NASA PDS. https://doi.org/10.17189/b2aj-cz96

## Notes

* `atomic_lines.db` is derived from the NIST Atomic Spectra Database; its compilation details are still to be documented (`data/README.md`).
* The code was written with AI assistance (Claude, Anthropic); see the manuscript's AI-assistance statement.

## License

Code: MIT (see `LICENSE`). The real spectra are not redistributed and remain under the terms of their original providers.

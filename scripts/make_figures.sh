#!/bin/bash
# Regenerates all figures from results/ (and, for Fig. 1, the dataset-1 cache).
set -e
cd "$(dirname "$0")/.."
python3 src/make_figures.py fig2 fig3 fig4 fig5 fig6 fig7
python3 src/fig8.py
python3 src/fig9.py

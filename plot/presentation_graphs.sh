#!/bin/bash

# Generuoja grafikus kursiniam.
# Pašalina senus grafikus jeigu tokie yra.

PLOT_DIR=../../bakalaurinis/pristatymas/plot
# PLOT_DIR=plot
BENCH_RESULTS_DIR=../results
# For putting two side-by-side.
STD_SCALE=0.45
# For one taking up full page width.
LARGE_SCALE=0.8
# FORMAT=png
FORMAT=pgf
PRE="python plot.py -p ${PLOT_DIR} -d ${BENCH_RESULTS_DIR} -f ${FORMAT} -s ${STD_SCALE}"

echo "Removing plot dir ${PLOT_DIR@Q}"
rm -Ir "${PLOT_DIR}"
mkdir "${PLOT_DIR}"

# Jeigu kas nors nepavyksta, baigiam.
set -eux

# CABC algoritmas
# cores_diff_gens
${PRE} -a CABC -k cores_diff_gens -t lin318 -c 1 2 4 6 8 --population-sizes 40 -e 2 --capital-ls 3


# qCABC algoritmas
# cores_diff_gens
${PRE} -a qCABC -k cores_diff_gens -t lin318 -c 1 2 4 6 8 --population-sizes 40 -e 8 --capital-ls 3


# PACO algoritmas
# cores_diff_gens
${PRE} -a PACO -k cores_diff_gens -c 1 2 4 6 8 -t lin318 --population-sizes 40 -e 32 --capital-ls 3


# Visi algoritmai
# cores-diff-algos
${PRE} -a PACO CABC qCABC -k cores_diff_algos -t lin318 -e 8 --capital-ls 2 --population-sizes 40

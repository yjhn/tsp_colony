#!/bin/bash

# Generuoja grafikus kursiniam.
# Pašalina senus grafikus jeigu tokie yra.

PLOT_DIR=../../bakalaurinis/plot
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
# L daro įtaką rezultatams, bet praktiškai random, todėl rezultatai rodomi su L = 2

# gens_diff_excg
# kroA100 ir eil51 grafikai neįdomūs
# Visur išskyrus lin318 D_m = 32 gerėja lėčiausiai, bet po to pralenkia kitus D_m.
# Su lin318, kuo mažesnis D_m, tuo geriau veikia.
${PRE} -a CABC -t lin318 -c 2 4 -e 2 4 8 32 --population-sizes 40 --capital-ls 3 -k gens_diff_excg --cond-y-top 3
${PRE} -a CABC -t kroA200 -c 2 4 -e 2 4 8 32 --population-sizes 40 --capital-ls 3 -k gens_diff_excg --cond-y-top 3
${PRE} -a CABC -t eil76 -c 2 4 -e 2 4 8 32 --population-sizes 40 --capital-ls 3 -k gens_diff_excg --cond-y-top 3


# cores_diff_test_cases
# Gal šito nedėsiu. Nieko įdomaus be to, kad rezultatai menkai gerėja su daugiau branduolių.
# TODO: grafikas, parodantis kartas ir branduolius su fiksuotu L.


# cores_diff_gens
# Šitas parodo, kad daugiau branduolių padeda žymiai greičiau gerėti iš pradžių, bet
# optimalaus sprendinio surasti iš esmės nepadeda.
${PRE} -a CABC -k cores_diff_gens -t lin318 -c 1 2 4 6 8 --population-sizes 40 -e 2 --capital-ls 3
# Su mažesniais miestų skaičiais labai mažai naudos iš > 2 branduolių.
${PRE} -a CABC -k cores_diff_gens -t kroA200 -c 1 2 4 6 8 --population-sizes 40 -e 32 --capital-ls 2
# Galima parodyti ir kroA100 čia, bet jis neįdomus, nes per lengvas.
${PRE} -a CABC -k cores_diff_gens -t kroA100 -c 1 2 4 6 8 --population-sizes 40 -e 32 --capital-ls 2
# eil76 elgiasi keistai: du branduoliai duoda prastesnius rezultatus nei 1
# eil51 rezultatai panašūs
${PRE} -a CABC -k cores_diff_gens -t eil76 -c 1 2 4 6 8 --population-sizes 40 -e 32 --capital-ls 2


# gens_diff_popsizes
# Kaip ir galima tikėtis, didesnė populiacija duoda geresnius rezultatus.
# Tačiau ne visada. Kartais su skirtingais L rezultatai apsiverčia.
${PRE} -a CABC -k gens_diff_popsizes -t lin318 d198 -c 6 8 --population-sizes 20 40 80 -e 8 --capital-ls 2 --cond-y-top 5


# cores_diff_popsizes
${PRE} -a CABC -k cores_diff_popsizes -t lin318 d198 -c 1 2 4 6 8 --population-sizes 20 40 80 -e 8 32 --capital-ls 2


# gens_diff_cls
# L įtaka rezultatams atsitiktinė.
${PRE} -a CABC -k gens_diff_cls -t lin318 d198 -c 6 --population-sizes 40 -e 8 --capital-ls 2 3 4
${PRE} -a CABC -k gens_diff_cls -t eil76 -c 1 4 --population-sizes 40 -e 8 --capital-ls 2 3 4 --cond-y-top 3


# cores_diff_cls
# Vėlgi, L įtaka atsitiktinė.
${PRE} -a CABC -k cores_diff_cls -t lin318 -c 1 2 4 6 8 --population-sizes 40 -e 8 --capital-ls 2 3 4
${PRE} -a CABC -k cores_diff_cls -t kroA200 d198 eil76 -c 1 2 4 6 8 --population-sizes 40 -e 32 --capital-ls 2 3 4


# Išvados: gal reikėjo testuoti ir su daugiau miestų? Nes su mažiau miestų rezultatai beveik random.


# qCABC algoritmas

# gens_diff_excg
${PRE} -a qCABC -t kroA200 -c 2 -e 2 4 8 32 --population-sizes 40 --capital-ls 2 -k gens_diff_excg --cond-y-top 3
${PRE} -a qCABC -t eil76 -c 2 8 -e 2 4 8 32 --population-sizes 40 --capital-ls 2 -k gens_diff_excg --cond-y-top 3
${PRE} -a qCABC -t kroA100 -c 8 -e 2 4 8 32 --population-sizes 40 --capital-ls 2 -k gens_diff_excg --cond-y-top 3

# cores_diff_gens
${PRE} -a qCABC -k cores_diff_gens -t lin318 -c 1 2 4 6 8 --population-sizes 40 -e 8 --capital-ls 3
${PRE} -a qCABC -k cores_diff_gens -t eil51 -c 1 2 4 6 8 --population-sizes 40 -e 32 --capital-ls 4
${PRE} -a qCABC -k cores_diff_gens -t d198 -c 1 2 4 6 8 --population-sizes 40 -e 32 --capital-ls 4
${PRE} -a qCABC -k cores_diff_gens -t kroA200 -c 1 2 4 6 8 --population-sizes 40 -e 32 --capital-ls 2

# gens_diff_popsizes
${PRE} -a qCABC -k gens_diff_popsizes -t eil76 -c 1 8 --population-sizes 20 40 80 -e 8 --capital-ls 2 --cond-y-top 5

# gens_diff_cls
${PRE} -a qCABC -k gens_diff_cls -t eil76 kroA200 -c 8 --population-sizes 40 -e 32 --capital-ls 2 3 4
${PRE} -a qCABC -k gens_diff_cls -t lin318 -c 6 --population-sizes 40 -e 8 32 --capital-ls 2 3 4 --cond-y-top 3

# cores_diff_cls
${PRE} -a qCABC -k cores_diff_cls -t lin318 eil76 -c 1 2 4 6 8 --population-sizes 40 -e 32 --capital-ls 2 3 4




# PACO algoritmas
${PRE} -a PACO -c 2 6 -e 2 4 8 32 --capital-ls 2 -k gens_diff_excg


# cores_diff_test_cases
${PRE} -a PACO -c 1 2 4 6 8 -e 8 32 --capital-ls 2 -k cores_diff_test_cases


# cores_diff_gens
${PRE} -a PACO -k cores_diff_gens -c 1 2 4 6 8 --population-sizes 40 -e 32 --capital-ls 3


# Visi algoritmai
# cores-diff-algos
${PRE} -a PACO CABC qCABC -k cores_diff_algos -e 8 32 --capital-ls 2 --population-sizes 40
# ${PRE} -a PACO CABC qCABC -k cores_diff_algos -t eil51 eil76 kroA100 -e 8 32 --capital-ls 2 --population-sizes 20
# ${PRE} -a PACO CABC qCABC -k cores_diff_algos -t eil51 eil76 -e 8 32 --capital-ls 2 --population-sizes 80

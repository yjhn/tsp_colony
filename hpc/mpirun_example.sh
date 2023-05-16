#!/bin/bash
#SBATCH --job-name=paco_2t
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -p main
#SBATCH --time 0-10:00:00
#SBATCH -n2
module load openmpi
mpirun target/release-opt/tsp_colony -f data/eil51.tsp data/eil76.tsp data/kroA100.tsp data/kroA200.tsp data/d198.tsp data/lin318.tsp --dup skip --max-iterations 4000 --algo paco --alphas 1 --betas 2 --ks 16 --lowercase-qs 3 --ros 0.1

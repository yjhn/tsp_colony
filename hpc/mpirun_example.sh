#!/bin/bash
#SBATCH --job-name=example
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -p main
#SBATCH --time 0-20:00:00
#SBATCH -n2
module load openmpi
mpirun target_pop/release/salesman -f data/att532.tsp -b 5 -a CgaThreeOpt --benchmark -m 500 -p 2 8 32 -e 4 --skip-duplicates

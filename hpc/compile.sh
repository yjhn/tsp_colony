#!/usr/bin/env bash

set -eux

module load openmpi
C_INCLUDE_PATH=/usr/lib/gcc/x86_64-linux-gnu/9/include"${C_INCLUDE_PATH+:}${C_INCLUDE_PATH-}" LIBCLANG_PATH="/scratch/lustre/home/${USER}/clang_stuff/usr/lib/llvm-12/lib" cargo build --profile release-opt

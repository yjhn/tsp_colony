#!/usr/bin/env sh

set -eu

results_dir="$(dirname $(realpath $0))/hpc_results"

echo "Removing write permission from destination files"
# Remove write permission on existing results files, this way scp will not overwrite them.
# Idea from: https://unix.stackexchange.com/a/51932
chmod -w "${results_dir}"/*.json

set +e -x

# Silence scp complaints that it is unable to write some files.
scp -r 'hpc.mif.vu.lt:bak/results/*' "${results_dir}" 3>&1 1>&2 2>&3 3>&- | grep -vx "scp: open local .*: Permission denied."

set -e +x

echo "Restoring write permission for destination files"
# Restore write permissions.
chmod +w "${results_dir}"/*.json

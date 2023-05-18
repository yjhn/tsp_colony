#!/usr/bin/env sh

set -eu

echo "Removing write permission fro mdestionation files"
# Remove write permission on existing results files, this way scp will not overwrite them.
# Idea from: https://unix.stackexchange.com/a/51932
chmod -w hpc_results/*.json

set +e -x

# Silence scp complaints that it is unable to write some files.
scp -r 'hpc.mif.vu.lt:bak/results/*' hpc_results

set -e +x

echo "Restoring write permission for destination files"
# Restore write permissions.
chmod +w hpc_results/*.json

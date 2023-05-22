#!/usr/bin/env sh

set -eu

results_dir_name="results"
# Get results dir relative to the script's location, not to cwd.
results_dir="$(dirname $(realpath $0))/${results_dir_name}"

echo "Removing write permission from destination files"
# Remove write permission on existing results files, this way scp will not overwrite them.
# Idea from: https://unix.stackexchange.com/a/51932
chmod -w "${results_dir}"/*.json

set +e -x

# Silence scp complaints that it is unable to write some files.
# Filter output lines by switching stdout and stderr: https://stackoverflow.com/a/2381643
scp -r 'hpc.mif.vu.lt:bak/results/*' "${results_dir}" 3>&1 1>&2 2>&3 3>&- | grep -vx "scp: open local .*: Permission denied."

set -e +x

echo "Restoring write permission for destination files"
# Restore write permissions.
chmod +w "${results_dir}"/*.json

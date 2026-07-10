#!/usr/bin/env bash
# Check whether test labels actually exist in the weak-lensing data directory.
# WeakLensingDataset (kan/data/datasets.py) currently hardcodes test_label=None
# and only loads test_kappa_file ("..._noisy_test.npy"). This script looks for
# a matching label file so we can tell whether that None is actually justified.
#
# Pure POSIX tools only (find, ls, strings/head) - no python/numpy needed, so
# it runs directly on the cluster login node. It can still peek at .npy
# headers because the numpy header is a plain ASCII dict right after the
# magic bytes.
set -uo pipefail

DATA_DIR="${1:-/cluster/projects/ska/weak-lensing/data}"

hr() { printf '%.0s-' {1..70}; echo; }

echo "Looking in: $DATA_DIR"
if [[ ! -d "$DATA_DIR" ]]; then
    echo "Directory not found (are you on the login node? did you mount/cd correctly?)."
    exit 1
fi
hr

echo "== Full directory listing =="
ls -la "$DATA_DIR"
hr

echo "== Known files expected by WeakLensingDataset =="
for f in "WIDE12H_bin2_2arcmin_kappa.npy" "label.npy" \
         "WIDE12H_bin2_2arcmin_kappa_noisy_test.npy" \
         "WIDE12H_bin2_2arcmin_mask.npy"; do
    if [[ -f "$DATA_DIR/$f" ]]; then
        printf "  [FOUND] %-50s %s\n" "$f" "$(ls -lh "$DATA_DIR/$f" | awk '{print $5}')"
    else
        printf "  [MISSING] %-50s\n" "$f"
    fi
done
hr

echo "== Any file with 'test' in the name =="
find "$DATA_DIR" -maxdepth 1 -type f -iname "*test*" -print
hr

echo "== Candidates that look like TEST labels (both 'label' and 'test' in name) =="
CANDIDATES=$(find "$DATA_DIR" -maxdepth 1 -type f \( -iname "*label*test*" -o -iname "*test*label*" \) -print)
if [[ -n "$CANDIDATES" ]]; then
    echo "$CANDIDATES"
else
    echo "(none found by naming convention)"
fi
hr

echo "== Peeking at .npy headers for label-like files (shape/dtype, no numpy needed) =="
for f in $(find "$DATA_DIR" -maxdepth 1 -type f -iname "*label*.npy"); do
    echo "--- $f ---"
    ls -lh "$f"
    head -c 200 "$f" | strings | head -3
    echo
done
hr

echo "== Summary =="
if [[ -n "$CANDIDATES" ]]; then
    echo "Possible test-label file(s) found - inspect the header output above and,"
    echo "if the shape matches (Nsys_test, n_targets) like label.npy does for train/val,"
    echo "update WeakLensingDataset to load it instead of hardcoding test_label=None."
else
    echo "No file matching a test-label naming pattern was found next to"
    echo "$DATA_DIR/WIDE12H_bin2_2arcmin_kappa_noisy_test.npy."
    echo "This supports test_label=None being correct - but check the full listing"
    echo "above in case the file uses a different naming convention."
fi

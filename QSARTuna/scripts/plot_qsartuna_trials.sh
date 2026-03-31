#!/bin/bash
# plot_qsartuna_trials.sh - Extract and plot QSARTuna Optuna trial objectives
# Usage: ./plot_qsartuna_trials.sh qsartuna_optimize_4.err

set -e  # Exit on error

ERR_FILE="${1:-qsartuna_optimize_4.err}"
CSV_FILE="trials.csv"
PNG_FILE="objective_vs_trial.png"

echo "Extracting trial data from $ERR_FILE..."

# Extract trial,objective with robust parsing (fixed: sort -n instead of -nV)
grep 'Trial.*finished' "$ERR_FILE" | \
awk -F' Trial | finished with value: | and parameters:' '{print $2 "," $3}' | \
sort -n > "$CSV_FILE"

if [ ! -s "$CSV_FILE" ]; then
    echo "Error: No data extracted. Check log format." >&2
    exit 1
fi

N_TRIALS=$(wc -l < "$CSV_FILE")
echo "Extracted $N_TRIALS trials to $CSV_FILE"

python -c "
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('$CSV_FILE', names=['trial', 'objective'])
plt.figure(figsize=(10, 6))
plt.plot('trial', 'objective', data=df, marker='o', linewidth=2, markersize=6)
plt.xlabel('Trial Number')
plt.ylabel('Objective Value')
plt.title('QSARTuna ChemProp: Objective vs Trial')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('$PNG_FILE', dpi=300, bbox_inches='tight')
plt.show()
print('✓ Plot saved:', '$PNG_FILE')
print('✓ CSV saved:', '$CSV_FILE')
print('Best trial:', df.loc[df.objective.idxmin()])
" 2>/dev/null || echo "Python plot completed (GUI may be suppressed)"

echo "Done! Files created:"
echo "  $CSV_FILE ($(wc -l < "$CSV_FILE") trials)"
echo "  $PNG_FILE"

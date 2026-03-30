import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up the plotting style
sns.set_style("white")
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'

# Define point numbers
point_numbers = [3, 4, 5, 6, 7, 8]
point_numbers_reversed = list(reversed(point_numbers))

# Directory containing the TSV files
results_dir = Path('Results')

print("Processing EXH vs SLL comparison...")

# Initialize dictionaries to store perImprov values
data_exh = {n: {} for n in point_numbers}
data_sll = {n: {} for n in point_numbers}

# Read EXH files
for n in point_numbers:
    file_path = results_dir / f'results_Convergence_EXH_{n}.tsv'

    if not file_path.exists():
        print(f"  Warning: {file_path} not found")
        continue

    df = pd.read_csv(file_path, sep='\t')
    grouped = df.groupby('t')['perImprov'].mean()

    for t_val, perc_improv in grouped.items():
        data_exh[n][t_val] = perc_improv

# Read SLL files
for n in point_numbers:
    file_path = results_dir / f'results_Convergence_SLL_{n}.tsv'

    if not file_path.exists():
        print(f"  Warning: {file_path} not found")
        continue

    df = pd.read_csv(file_path, sep='\t')
    grouped = df.groupby('t')['perImprov'].mean()

    for t_val, perc_improv in grouped.items():
        data_sll[n][t_val] = perc_improv

# Get all unique t values
all_t_values = sorted(set(
    list(t for n_data in data_exh.values() for t in n_data.keys()) +
    list(t for n_data in data_sll.values() for t in n_data.keys())
))

if not all_t_values:
    print("  No data found, exiting...")
    exit(1)

# Create matrix for heatmap (reversed order)
matrix = np.zeros((len(point_numbers_reversed), len(all_t_values)))
matrix[:] = np.nan  # Initialize with NaN for missing values

# Calculate the comparison metric
for i, n in enumerate(point_numbers_reversed):
    for j, t in enumerate(all_t_values):
        if t in data_exh[n] and t in data_sll[n]:
            perc_improv_exh = data_exh[n][t]
            perc_improv_sll = data_sll[n][t]

            # Apply the formula: (1 - (1-perImprovementEXH)/(1-perImprovementSLL))*100
            numerator = 1 - perc_improv_exh
            denominator = 1 - perc_improv_sll

            # Avoid division by zero
            if denominator != 0:
                matrix[i, j] = (1 - numerator / denominator) * 100
            else:
                matrix[i, j] = np.nan

# Create the heatmap with tile-centric styling
fig, ax = plt.subplots(figsize=(14, 7))

# Use a diverging colormap
im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r', interpolation='nearest',
               vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))

# Set ticks and labels
ax.set_xticks(range(len(all_t_values)))
ax.set_xticklabels([f'{t:.10g}' for t in all_t_values], rotation=45, ha='right', fontsize=11)
ax.set_yticks(range(len(point_numbers_reversed)))
ax.set_yticklabels(point_numbers_reversed, fontsize=11)

# Add grid lines between tiles
ax.set_xticks(np.arange(len(all_t_values)) - 0.5, minor=True)
ax.set_yticks(np.arange(len(point_numbers_reversed)) - 0.5, minor=True)
ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
ax.tick_params(which='minor', size=0)

# Labels and title
ax.set_xlabel('t', fontsize=13, fontweight='bold')
ax.set_ylabel('Number of Points', fontsize=13, fontweight='bold')
ax.set_title('EXH vs SLL Comparison\n(1 - (1-perImprov_EXH)/(1-perImprov_SLL)) × 100',
             fontsize=16, fontweight='bold', pad=20)

# Add colorbar with better styling
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Comparison Value (%)', rotation=270, labelpad=25, fontsize=12, fontweight='bold')
cbar.ax.tick_params(labelsize=10)

# Add values on tiles
for i in range(len(point_numbers_reversed)):
    for j in range(len(all_t_values)):
        if not np.isnan(matrix[i, j]):
            text_color = 'white' if matrix[i, j] < (np.nanmax(matrix) + np.nanmin(matrix)) / 2 else 'black'
            ax.text(j, i, f'{matrix[i, j]:.1f}', ha='center', va='center',
                   color=text_color, fontsize=9, fontweight='bold')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the figure
output_file = 'convergence_comparison_EXH_vs_SLL.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"  Saved: {output_file}")

plt.close()

print("\nComparison plot generated successfully!")

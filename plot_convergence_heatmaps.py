import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up NeurIPS-style plotting parameters
plt.rcParams.update({
    'font.size': 9,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'text.usetex': False,  # Set to True if you have LaTeX installed
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.5,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.0,
})
sns.set_style("white")

# Define methods and point numbers
methods = ['EXH', 'NJ', 'SLL']
method_labels = {'EXH': 'RHS', 'NJ': 'NJ', 'SLL': 'HS'}
point_numbers = [3, 4, 5, 6, 7, 8, 9, 10]
point_numbers_reversed = list(reversed(point_numbers))

# Directory containing the TSV files
results_dir = Path('Results')

# First pass: collect all data to find global min/max
all_matrices = {}
all_t_values_per_method = {}

for method in methods:
    print(f"Loading {method} data...")
    data_dict = {n: {} for n in point_numbers}

    # Read all files for this method
    for n in point_numbers:
        file_path = results_dir / f'results_Convergence_{method}_{n}.tsv'
        if file_path.exists():
            df = pd.read_csv(file_path, sep='\t')
            grouped = df.groupby('t')['perImprov'].mean() * 100
            for t_val, perc_improv in grouped.items():
                data_dict[n][t_val] = perc_improv

    # Get all unique t values across all files
    all_t_values = sorted(set(t for n_data in data_dict.values() for t in n_data.keys()))
    all_t_values_per_method[method] = all_t_values

    # Create matrix for heatmap (reversed order)
    matrix = np.zeros((len(point_numbers_reversed), len(all_t_values)))
    matrix[:] = np.nan

    for i, n in enumerate(point_numbers_reversed):
        for j, t in enumerate(all_t_values):
            if t in data_dict[n]:
                matrix[i, j] = data_dict[n][t]

    all_matrices[method] = matrix

# Find global min/max across all methods for consistent normalization
global_min = min(np.nanmin(mat) for mat in all_matrices.values())
global_max = max(np.nanmax(mat) for mat in all_matrices.values())

print(f"\nGlobal value range: [{global_min:.2f}, {global_max:.2f}]")
for method in methods:
    mat = all_matrices[method]
    print(f"  {method}: [{np.nanmin(mat):.2f}, {np.nanmax(mat):.2f}]")

# Second pass: create plots with consistent normalization
for method in methods:
    print(f"Creating plot for {method}...")

    # Get the precomputed matrix and t values
    matrix = all_matrices[method]
    all_t_values = all_t_values_per_method[method]

    if len(all_t_values) == 0:
        print(f"  No data found for {method}, skipping...")
        continue

    # Create the heatmap with NeurIPS-appropriate styling
    fig, ax = plt.subplots(figsize=(3.5, 3.2))

    # Use consistent PuOr colormap for all methods with global normalization centered at 0
    # 'PuOr' (Purple-Orange): purple = negative/worse, white = neutral, orange = positive/better
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=global_min, vcenter=0, vmax=global_max)
    cmap = 'PuOr'
    im = ax.imshow(matrix, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')

    # Set ticks and labels
    # Format t values to show 1-1e-X for values very close to 1
    def format_t_label(t):
        if t >= 0.999:
            exp = int(np.floor(np.log10(1 - t)))
            return f'$1-10^{{{exp}}}$'
        else:
            return f'{t:.2g}'

    ax.set_xticks(range(len(all_t_values)))
    ax.set_xticklabels([format_t_label(t) for t in all_t_values], rotation=45, ha='right')
    ax.set_yticks(range(len(point_numbers_reversed)))
    ax.set_yticklabels(point_numbers_reversed)

    # Add grid lines between tiles
    ax.set_xticks(np.arange(len(all_t_values)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(point_numbers_reversed)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1.5)
    ax.tick_params(which='minor', size=0)
    ax.tick_params(which='major', length=2, width=0.5)

    # Labels and title
    ax.set_xlabel('$t$', fontsize=9)
    ax.set_ylabel('Number of Clusters', fontsize=9)
    ax.set_title(f'{method_labels[method]}', fontsize=10, pad=8)

    # Add colorbar with better styling
    cbar = plt.colorbar(im, ax=ax, fraction=0.08, pad=0.03)
    cbar.set_label('Reduction over MST (%)', rotation=270, labelpad=12, fontsize=8)

    # Set better colorbar ticks - consistent across all plots
    if global_min < 0 < global_max:
        # Add ticks around 0 if range includes negatives
        negative_ticks = np.linspace(global_min, 0, 3)
        positive_ticks = np.linspace(0, global_max, 5)[1:]
        tick_vals = np.concatenate([negative_ticks, positive_ticks])
    else:
        tick_vals = np.linspace(global_min, global_max, 7)

    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels([f'{v:.1f}' for v in tick_vals])
    cbar.ax.tick_params(labelsize=7, width=0.5, length=2)
    cbar.outline.set_linewidth(0.5)

    # Add values on tiles
    for i in range(len(point_numbers_reversed)):
        for j in range(len(all_t_values)):
            if not np.isnan(matrix[i, j]):
                # PuOr diverging: use white for extreme values (dark purple or dark orange)
                # Dark colors at extremes, light colors near 0
                if matrix[i, j] < global_min * 0.4 or matrix[i, j] > global_max * 0.4:
                    text_color = 'white'
                else:
                    text_color = 'black'
                ax.text(j, i, f'{matrix[i, j]:.1f}', ha='center', va='center',
                       color=text_color, fontsize=6)

    # Adjust layout to prevent label cutoff
    plt.tight_layout(pad=0.3)

    # Save the figure
    output_file = f'convergence_heatmap_{method}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.05)
    pdf_file = f'convergence_heatmap_{method}.pdf'
    plt.savefig(pdf_file, bbox_inches='tight', pad_inches=0.05)
    print(f"  Saved: {output_file} and {pdf_file}")

    plt.close()

print("\nAll plots generated successfully!")

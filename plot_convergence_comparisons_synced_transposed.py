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

# Define point numbers
point_numbers = [3, 4, 5, 6, 7, 8, 9, 10]

# Directory containing the TSV files
results_dir = Path('Results')

print("Processing EXH comparisons with NJ and SLL (transposed)...")

# Initialize dictionaries to store perImprov values
data_exh = {n: {} for n in point_numbers}
data_nj = {n: {} for n in point_numbers}
data_sll = {n: {} for n in point_numbers}

# Read EXH files
for n in point_numbers:
    file_path = results_dir / f'results_Convergence_EXH_{n}.tsv'
    if file_path.exists():
        df = pd.read_csv(file_path, sep='\t')
        grouped = df.groupby('t')['perImprov'].mean()
        for t_val, perc_improv in grouped.items():
            data_exh[n][t_val] = perc_improv

# Read NJ files
for n in point_numbers:
    file_path = results_dir / f'results_Convergence_NJ_{n}.tsv'
    if file_path.exists():
        df = pd.read_csv(file_path, sep='\t')
        grouped = df.groupby('t')['perImprov'].mean()
        for t_val, perc_improv in grouped.items():
            data_nj[n][t_val] = perc_improv

# Read SLL files
for n in point_numbers:
    file_path = results_dir / f'results_Convergence_SLL_{n}.tsv'
    if file_path.exists():
        df = pd.read_csv(file_path, sep='\t')
        grouped = df.groupby('t')['perImprov'].mean()
        for t_val, perc_improv in grouped.items():
            data_sll[n][t_val] = perc_improv

# Get all unique t values
all_t_values = sorted(set(
    list(t for n_data in data_exh.values() for t in n_data.keys()) +
    list(t for n_data in data_nj.values() for t in n_data.keys()) +
    list(t for n_data in data_sll.values() for t in n_data.keys())
))

if not all_t_values:
    print("  No data found, exiting...")
    exit(1)

# Create matrices for both comparisons (TRANSPOSED: t values on y-axis, point numbers on x-axis)
matrix_sll = np.zeros((len(all_t_values), len(point_numbers)))
matrix_sll[:] = np.nan

matrix_nj = np.zeros((len(all_t_values), len(point_numbers)))
matrix_nj[:] = np.nan

# Calculate comparison metrics
for i, t in enumerate(all_t_values):
    for j, n in enumerate(point_numbers):
        # EXH vs SLL
        if t in data_exh[n] and t in data_sll[n]:
            perc_improv_exh = data_exh[n][t]
            perc_improv_sll = data_sll[n][t]
            numerator = 1 - perc_improv_exh
            denominator = 1 - perc_improv_sll
            if denominator != 0:
                matrix_sll[i, j] = (1 - numerator / denominator) * 100

        # EXH vs NJ
        if t in data_exh[n] and t in data_nj[n]:
            perc_improv_exh = data_exh[n][t]
            perc_improv_nj = data_nj[n][t]
            numerator = 1 - perc_improv_exh
            denominator = 1 - perc_improv_nj
            if denominator != 0:
                matrix_nj[i, j] = (1 - numerator / denominator) * 100

# Find global min and max for synchronized color scale
global_min = min(np.nanmin(matrix_sll), np.nanmin(matrix_nj))
global_max = max(np.nanmax(matrix_sll), np.nanmax(matrix_nj))

print(f"Global color scale: min={global_min:.2f}, max={global_max:.2f}")
print(f"SLL range: [{np.nanmin(matrix_sll):.2f}, {np.nanmax(matrix_sll):.2f}]")
print(f"NJ range: [{np.nanmin(matrix_nj):.2f}, {np.nanmax(matrix_nj):.2f}]")

# Method label mapping
method_labels = {'EXH': 'RHS', 'SLL': 'HS', 'NJ': 'NJ'}

# Function to create a comparison heatmap (NeurIPS paper ready, TRANSPOSED)
def create_comparison_plot(matrix, method_name, filename, vmin, vmax):
    # NeurIPS column width is ~3.25 inches, page width is ~6.875 inches
    # Use a width that fits well in a 2-column layout
    # Adjust figure size for transposed layout to make cells more square
    fig, ax = plt.subplots(figsize=(3.5, 2.0))

    # Use a colorblind-friendly and print-ready sequential colormap
    # Positive values = EXH is better (higher is better)
    # Using 'PuOr' (Purple-Orange) diverging, centered at 0
    # Purple = negative (baseline better), White = neutral, Orange = positive (EXH better)
    # This is accessible, prints well, and is colorblind-friendly
    from matplotlib.colors import TwoSlopeNorm

    # Center the colormap at 0 for intuitive interpretation
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    cmap = 'PuOr'

    im = ax.imshow(matrix, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')

    # Set ticks and labels with NeurIPS-appropriate sizes
    # Format t values to show 1-1e-X for values very close to 1
    def format_t_label(t):
        if t >= 0.999:
            exp = int(np.floor(np.log10(1 - t)))
            return f'$1-10^{{{exp}}}$'
        else:
            return f'{t:.2g}'

    # TRANSPOSED: x-axis is point numbers, y-axis is t values
    ax.set_xticks(range(len(point_numbers)))
    ax.set_xticklabels(point_numbers)
    ax.set_yticks(range(len(all_t_values)))
    ax.set_yticklabels([format_t_label(t) for t in all_t_values])

    # Add grid lines between tiles
    ax.set_xticks(np.arange(len(point_numbers)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(all_t_values)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1.5)
    ax.tick_params(which='minor', size=0)
    ax.tick_params(which='major', length=2, width=0.5)

    # Labels and title (concise for paper, TRANSPOSED)
    ax.set_xlabel('Number of Clusters', fontsize=9)
    ax.set_ylabel('$t$', fontsize=9)
    ax.set_title(f'{method_labels["EXH"]} vs {method_labels[method_name]}', fontsize=10, pad=8)

    # Add colorbar with better styling
    cbar = plt.colorbar(im, ax=ax, fraction=0.08, pad=0.03)
    cbar.set_label('Rel. Improvement (%)', rotation=270, labelpad=12, fontsize=8)

    # Set explicit colorbar ticks for clarity
    # Create more ticks with better distribution
    if vmin < 0 < vmax:
        # Asymmetric range: add more ticks on the negative side
        negative_ticks = np.linspace(vmin, 0, 3)  # e.g., -0.4, -0.2, 0.0
        positive_ticks = np.linspace(0, vmax, 7)[1:]  # e.g., 7, 14, 21, 28, 35, 43
        tick_vals = np.concatenate([negative_ticks, positive_ticks])
    else:
        tick_vals = np.linspace(vmin, vmax, 9)

    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels([f'{v:.1f}' for v in tick_vals])

    cbar.ax.tick_params(labelsize=7, width=0.5, length=2)
    cbar.outline.set_linewidth(0.5)

    # Add values on tiles (smaller font for paper, TRANSPOSED: i is t index, j is point number index)
    for i in range(len(all_t_values)):
        for j in range(len(point_numbers)):
            if not np.isnan(matrix[i, j]):
                # PuOr diverging: use white for extreme values (dark purple or dark orange)
                # Dark colors at extremes, light colors near 0
                if matrix[i, j] < vmin * 0.4 or matrix[i, j] > vmax * 0.4:
                    text_color = 'white'
                else:
                    text_color = 'black'
                ax.text(j, i, f'{matrix[i, j]:.1f}', ha='center', va='center',
                       color=text_color, fontsize=6)

    # Adjust layout to prevent label cutoff
    plt.tight_layout(pad=0.3)

    # Save the figure
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.05)
    # Also save as PDF (preferred for papers)
    pdf_filename = filename.replace('.png', '.pdf')
    plt.savefig(pdf_filename, bbox_inches='tight', pad_inches=0.05)
    print(f"  Saved: {filename} and {pdf_filename}")
    plt.close()

# Create both plots with synchronized color scales
create_comparison_plot(matrix_sll, 'SLL', 'convergence_comparison_EXH_vs_SLL_transposed.png', global_min, global_max)
create_comparison_plot(matrix_nj, 'NJ', 'convergence_comparison_EXH_vs_NJ_transposed.png', global_min, global_max)

print("\nTransposed comparison plots generated successfully with synchronized color scales!")

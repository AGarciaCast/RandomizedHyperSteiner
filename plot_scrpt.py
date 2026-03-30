import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from pathlib import Path





FIG_PATH =  Path("./Figures/Results")

# Set seaborn style
sns.set_theme(style="darkgrid")

# Set publication-quality style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'text.usetex': False,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'lines.linewidth': 2.5,
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.3,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
})

# Define the functions
def f(n):
    return np.floor(2 * np.sqrt(n) - 1)

def identity(n):
    return np.floor(n)

def constant_one(n):
    return np.ones_like(n)

# Create x values - INTEGER VALUES ONLY
n = np.arange(1, 21)  # This gives [1, 2, 3, ..., 20]

# Calculate y values
y_step = f(n)
y_linear = identity(n)
y_constant = constant_one(n)

# Flexoki 2 color palette
color_ours = '#D0A215'      # RHS (Yellow from Flexoki)
color_beasly = '#8B7EC8'    # Purple from Flexoki 2
color_laarhoven = '#CE5D97' # Magenta from Flexoki 2

# Create the plot with publication-quality dimensions
fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

# Set background transparency
ax.patch.set_alpha(0.5)

# Plot the functions with markers
ax.plot(n, y_linear, linewidth=2.5, color=color_beasly, label='Beasly et al.',
        alpha=0.9, marker='o', markersize=8)
ax.plot(n, y_constant, linewidth=3.5, color=color_laarhoven, label='Laarhoven et al.',
        alpha=1.0, linestyle='-', dash_capstyle='round', marker='s', markersize=8)
ax.plot(n, y_step, linewidth=2.5, color=color_ours, label='RHS (Ours)',
        alpha=0.9, marker='^', markersize=8)

# Labels
ax.set_xlabel(r'Iteration ($n$)', fontsize=17)
ax.set_ylabel(r'Number of Expansions', fontsize=17)

# Legend with publication styling
legend = ax.legend(fontsize=11, loc='upper left', frameon=True,
                   fancybox=False, edgecolor='black', framealpha=0.95)
# Make legend text darker
for text in legend.get_texts():
    text.set_color('black')

# Set axis limits
ax.set_xlim(0.5, 20.5)
ax.set_ylim(0.5, max(y_linear) + 1)

# Force integer ticks on both axes
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# Set y-ticks to start at 1
y_max = int(max(y_linear) + 1)
ax.set_yticks(range(1, y_max + 1))

plt.tight_layout(pad=0.3)
plt.savefig(FIG_PATH/'expansion_curve.pdf', dpi=300, bbox_inches='tight', 
            format='pdf', backend='pdf')

plt.show()
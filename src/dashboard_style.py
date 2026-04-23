"""
TacticalVision AI: Premium Dashboard Design System
Shared constants and utilities for all dashboard visualizations.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ── Color System ─────────────────────────────────────────────────────────
BG       = '#0D1117'
CARD     = '#161B22'
BORDER   = '#30363D'
TEXT     = '#E6EDF3'
TEXT_SEC = '#8B949E'
GRID     = '#21262D'

# Accent colors
BLUE      = '#58A6FF'
BLUE_DIM  = '#1F6FEB'
GREEN     = '#3FB950'
GREEN_DIM = '#238636'
PURPLE    = '#BC8CFF'
PURPLE_DIM= '#8957E5'
ORANGE    = '#F0883E'
ORANGE_DIM= '#BD561D'
RED       = '#FF7B72'
RED_DIM   = '#DA3633'
CYAN      = '#76E3EA'
YELLOW    = '#E3B341'
PINK      = '#F778BA'

# Sequential palettes (light → dark) for bar charts
PALETTE_BLUE   = ['#7EBFFF','#58A6FF','#3B9BE0','#2178B5','#1F5A8C','#1F3D5C']
PALETTE_GREEN  = ['#6CD76C','#3FB950','#2EA44F','#23803B','#1F5C2F','#1A3D23']
PALETTE_PURPLE = ['#D4B5FF','#BC8CFF','#8957E5','#6B3FA0','#4B2D7A','#2D1B4E']
PALETTE_ORANGE = ['#FFB070','#F0883E','#BD561D','#9A5500','#6B3A00','#3D2200']
PALETTE_RED    = ['#FFA198','#FF7B72','#DA3633','#B62324','#8B1A1A','#5C1010']
PALETTE_CYAN   = ['#B0F0F4','#76E3EA','#3FC7D0','#2A9DA5','#1B7A80','#0F4D52']


def setup_premium_style():
    """Configure matplotlib with the TacticalVision premium dark theme."""
    plt.rcParams.update({
        'figure.facecolor':    BG,
        'figure.edgecolor':    BG,
        'savefig.facecolor':   BG,
        'savefig.edgecolor':   BG,
        'axes.facecolor':      CARD,
        'axes.edgecolor':      BORDER,
        'axes.labelcolor':     TEXT,
        'axes.titlecolor':     TEXT,
        'text.color':          TEXT,
        'xtick.color':         TEXT_SEC,
        'ytick.color':         TEXT_SEC,
        'grid.color':          GRID,
        'grid.alpha':          0.6,
        'grid.linestyle':      '--',
        'grid.linewidth':      0.5,
        'figure.titlesize':    22,
        'figure.titleweight':  'bold',
        'axes.titlesize':      14,
        'axes.titleweight':    'bold',
        'axes.titlepad':       12,
        'axes.labelsize':      12,
        'axes.labelpad':       8,
        'xtick.labelsize':     10,
        'ytick.labelsize':     10,
        'legend.facecolor':    CARD,
        'legend.edgecolor':    BORDER,
        'legend.fontsize':     10,
        'legend.labelcolor':   TEXT,
        'font.family':         'sans-serif',
        'font.sans-serif':     ['Segoe UI', 'Arial', 'DejaVu Sans'],
        'lines.linewidth':     2.0,
        'patch.edgecolor':     BORDER,
    })


def stat_box(ax, text, x=0.97, y=0.95, ha='right', va='top'):
    """Draw a glassmorphism-style stat annotation box."""
    props = dict(boxstyle='round,pad=0.6', facecolor=CARD,
                 edgecolor=BORDER, alpha=0.90)
    ax.text(x, y, text, transform=ax.transAxes, fontsize=10,
            ha=ha, va=va, bbox=props, color=TEXT_SEC,
            family='monospace')


def style_axis(ax, title=None, xlabel=None, ylabel=None):
    """Apply consistent styling to a single axis."""
    if title:
        ax.set_title(title, pad=12, fontsize=14, fontweight='bold', color=TEXT)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11, color=TEXT_SEC)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, color=TEXT_SEC)
    ax.tick_params(colors=TEXT_SEC, which='both')
    ax.grid(True, alpha=0.3, color=GRID, linestyle='--', linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(0.8)


def mean_line(ax, value, label_fmt="Mean: {:.3f}", color=CYAN, vertical=True):
    """Add a styled mean reference line."""
    fn = ax.axvline if vertical else ax.axhline
    fn(value, color=color, ls='--', lw=1.8, alpha=0.9, zorder=5,
       label=label_fmt.format(value))


def make_gradient_cmap(c1, c2, name='custom'):
    """Create a LinearSegmentedColormap between two hex colors."""
    return mcolors.LinearSegmentedColormap.from_list(name, [c1, c2], N=256)


def gradient_bar_colors(n, palette):
    """Generate n colors from a palette (interpolating if needed)."""
    cmap = mcolors.LinearSegmentedColormap.from_list('grad', palette, N=n)
    return [cmap(i / max(1, n - 1)) for i in range(n)]


def suptitle(fig, text, y=0.97):
    """Add a premium main title to the figure."""
    fig.suptitle(text, fontsize=22, fontweight='bold', color=TEXT, y=y,
                 fontfamily='sans-serif')

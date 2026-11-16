import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams['font.family'] = ['Inter', 'Segoe UI', 'DejaVu Sans']
cards = ['12 GB', '16 GB', '24 GB', '32 GB']
A = [None, 46.4, 71.2, 154.0]          # K/V in VRAM (selection)
C = [47.2, 62.1, 74.4, 144.0]          # K/V offloaded (mean of 3 runs where available)
blue, orange = '#2a78d6', '#eb6834'
ink, ink2, grid, surf = '#1f1f1e', '#5f5e58', '#e6e5e0', '#fcfcfb'
fig, ax = plt.subplots(figsize=(8, 4.4), dpi=200)
fig.patch.set_facecolor(surf); ax.set_facecolor(surf)
w, gap = 0.36, 0.02
for i, (a, c) in enumerate(zip(A, C)):
    for v, x, col in ((a, i - w / 2 - gap / 2, blue), (c, i + w / 2 + gap / 2, orange)):
        if v is None:
            ax.text(x, 4, 'does not\nfit', ha='center', va='bottom', fontsize=8.5, color=ink2)
            continue
        ax.add_patch(FancyBboxPatch((x - w / 2, 0), w, v, boxstyle='round,pad=0,rounding_size=0.04',
                                    mutation_aspect=40, linewidth=0, facecolor=col))
        ax.text(x, v + 3, f'{v:.0f}', ha='center', va='bottom', fontsize=9, color=ink)
ax.set_xlim(-0.6, 3.6); ax.set_ylim(0, 175)
ax.set_xticks(range(4), cards, fontsize=10, color=ink)
ax.set_ylabel('Decode speed (tokens/s)', fontsize=10, color=ink2)
ax.yaxis.grid(True, color=grid, linewidth=0.8); ax.set_axisbelow(True)
for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
ax.spines['bottom'].set_color('#bdbcb5'); ax.tick_params(axis='y', colors=ink2, labelsize=9, length=0)
ax.tick_params(axis='x', length=0)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=blue, label='K/V in VRAM'), Patch(color=orange, label='K/V offloaded to host memory')],
          loc='upper left', frameon=False, fontsize=9.5, labelcolor=ink)
ax.set_title('Qwen3.6-35B-A3B, 128K context: decode speed on simulated cards (RTX 5090 with a VRAM cap)',
             fontsize=10.5, color=ink, loc='left', pad=12)
fig.tight_layout()
import sys; fig.savefig(sys.argv[1] if len(sys.argv) > 1 else 'figures/vram_cap_decode_128k.png', facecolor=surf)

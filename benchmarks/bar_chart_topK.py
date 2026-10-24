import matplotlib.pyplot as plt
import seaborn as sns

# Stílus beállítása a modern, letisztult megjelenésért
sns.set_theme(style="whitegrid")

# Adatok
categories = [
    'Random baseline\n(Top-3 accuracy)',
    'TPOT with MultiRocket\nfingerprints',
    'TPOT with Signature\nfingerprints'
]
values = [0.6, 0.76, 0.657]
colors = ['#e1d5e7', '#2ECC71', "#E64922"]

# Diagram mérete és létrehozása
fig, ax = plt.subplots(figsize=(8, 4), dpi=300)

# Oszlopok kirajzolása
bars = ax.bar(categories, values, color=colors, width=0.45, edgecolor='none', zorder=3)

# Értékek feltüntetése az oszlopok tetején
for bar in bars:
    height = bar.get_height()
    ax.annotate(
        f'{height:.3f}',
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 6),
        textcoords="offset points",
        ha='center',
        va='bottom',
        fontsize=11,
        fontweight='bold',
        color='#2C3E50'
    )

# Címek és tengelyek beállítása
ax.set_ylabel('Top-3 accuracy', fontsize=12, labelpad=10, color='#2C3E50')
ax.set_ylim(0.5,0.9)

# Vonalak és keretek finomhangolása
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#BDC3C7')
ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax.xaxis.grid(False)

ax.tick_params(axis='x', labelsize=9, colors='#2C3E50')
ax.tick_params(axis='y', labelsize=10, colors='#2C3E50')

# Helykihasználás optimálása
plt.tight_layout()

# Mentés képfájlként
plt.savefig('bar_chart_topK.png', dpi=300, bbox_inches='tight')

# Megjelenítés
plt.show()
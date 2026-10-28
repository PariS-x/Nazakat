import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
})

conditions = ['A\n(Rigid+Std)', 'B\n(Soft+Std)', 'C\n(Rigid+Dev)', 'D\n(Soft+Dev)']
train = [20.0, 100.0, 0.0, 35.0]
gen = [11.67, 85.0, 0.0, 30.0]

x = np.arange(len(conditions))
width = 0.35

fig, ax = plt.subplots(figsize=(6.5, 3.2))
bars1 = ax.bar(x - width/2, train, width, label='Train objects', color='#4C72B0', edgecolor='black', linewidth=0.6)
bars2 = ax.bar(x + width/2, gen, width, label='Test objects (generalization)', color='#DD8452', edgecolor='black', linewidth=0.6)

for bars in (bars1, bars2):
    for b in bars:
        h = b.get_height()
        ax.annotate(f'{h:.1f}%', xy=(b.get_x() + b.get_width()/2, h),
                    xytext=(0, 2), textcoords='offset points',
                    ha='center', va='bottom', fontsize=7.5)

ax.set_ylabel('Grasp success rate (%)')
ax.set_xticks(x)
ax.set_xticklabels(conditions)
ax.set_ylim(0, 108)
ax.legend(loc='upper right', frameon=True, fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, linestyle=':', alpha=0.5)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('results_bar_chart.pdf', bbox_inches='tight')
print("saved")
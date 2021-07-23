# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
plt.style.use('seaborn')
plt.style.use('seaborn-talk')
plt.style.use('seaborn-colorblind')
# 'seaborn', 'seaborn-colorblind', 'seaborn-talk'
data = np.loadtxt('DiamondChartUNet.csv', delimiter=',')
params = ['Base Model', 'Normal map', 'Standard Deviation map',
          'Depth map', 'Reflectance map', 'UNet-512 + Normal map']

fig, ax = plt.subplots(1, figsize=(15, 15), subplot_kw=dict(polar=True))
# fig.set_color_cycle(sns.color_palette("mako", 5))
for i in range(6):
    ax.fill([0, np.pi*0.5, np.pi, np.pi*1.5],
            data[i], alpha=0.35, edgecolor='#000000', linewidth=1,
            zorder=1 if i == 5 else i)

ax.legend(params, loc='upper right')
ax.tick_params(labelsize=13)
for i in range(5):
    ax.plot([0, np.pi*0.5, np.pi, np.pi*1.5, 0],
            np.hstack((data[i], data[i, 0])), alpha=1,
            linewidth=0.2, color='black')
ax.grid(color='#AAAAAA')
ax.set_rgrids([0, 1], color='#FFFFFF')
ax.set_facecolor('#FFFFFF')
ax.spines['polar'].set_color('#222222')
# ax.tick_params(colors='#FFFFFF')
ax.set_thetagrids(np.degrees(
    [0, np.pi*0.5, np.pi, np.pi*1.5]), ['MSE', 'RER', 'SSIM', 'PSNR'])
# ax.plot([-1, 1], [0, 0], color=[0.6,0.6,0.6])
# ax.plot([0, 0], [-1, 1], color=[0.6,0.6,0.6])
# ax.annotate('Performance', [0, -1.05])
# ax.annotate('MSE', [1.02, 0])
# ax.annotate('SSIM', [0, 1.03])
# ax.annotate('PSNR', [-1.08, 0])
# plt.savefig('diamondChartUNet.png', dpi=150)

# %%
plt.style.use('seaborn')
plt.style.use('seaborn-talk')
plt.style.use('seaborn-colorblind')

data = np.loadtxt('DiamondChartSpeed.csv', delimiter=',')
params = ['Base Model', 'Normal map', 'Standard Deviation map',
          'Depth map', 'Reflectance map', 'UNet-512 + Normal map']

fig, ax = plt.subplots(1, figsize=(15, 15), subplot_kw=dict(polar=True))
# fig.set_color_cycle(sns.color_palette("mako", 5))
for i in range(6):
    ax.fill([np.pi*0.1, np.pi*0.5, np.pi*0.9, np.pi*1.3, np.pi*1.7],
            data[i], alpha=0.35, edgecolor='#000000', linewidth=1,
            zorder=0 if i == 5 else None)

ax.legend(params, loc='upper right')
ax.tick_params(labelsize=13)
for i in range(6):
    ax.plot([np.pi*0.1, np.pi*0.5, np.pi*0.9, np.pi*1.3, np.pi*1.7, np.pi*0.1],
            np.hstack((data[i], data[i, 0])), alpha=1,
            linewidth=0.2, color='black')
ax.grid(color='#AAAAAA')
ax.set_rgrids([0, 1], color='#FFFFFF')
ax.set_facecolor('#FFFFFF')
ax.spines['polar'].set_color('#222222')
# ax.tick_params(colors='#FFFFFF')
ax.set_thetagrids(np.degrees(
    [np.pi*0.1, np.pi*0.5, np.pi*0.9, np.pi*1.3, np.pi*1.7]),
    ['MSE', 'RER', 'SSIM', 'PSNR', 'Runtime'])
# ax.plot([-1, 1], [0, 0], color=[0.6,0.6,0.6])
# ax.plot([0, 0], [-1, 1], color=[0.6,0.6,0.6])
# ax.annotate('Performance', [0, -1.05])
# ax.annotate('MSE', [1.02, 0])
# ax.annotate('SSIM', [0, 1.03])
# ax.annotate('PSNR', [-1.08, 0])
plt.savefig('diamondChartRuntime.png', dpi=150)

# %%
# Logarithm figure generator
x = np.arange(0.00001, 3, 0.00001)
log = np.log10(x)
dLog = 1/(x*np.log(10))

fig, ax1 = plt.subplots(1, figsize=(10, 8))
ax1.plot(x, log, color='black', lw=3)
ax1.plot(x, dLog, color='black', ls='--')
# ax1.plot([-3, 3], [0, 0], color=[0.3,0.3,0.3])
# ax1.annotate('data hist', (0.05, 1))
ax1.set_xlim([-0.04, 1])
ax1.set_ylim([-3, 6])
ax1.grid(True)
ax1.legend(['Log(x)', 'd/dx (log(x))'], prop={'size': 30})

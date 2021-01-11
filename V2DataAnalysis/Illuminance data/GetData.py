# %%
import numpy as np

Name = '1,0,0'
# %%
data = np.genfromtxt(f'{Name}.ill', delimiter=' ')
data = np.delete(data, 3, 1)
np.savetxt(f'{Name}.gz', data, '%s')

# %%
print(np.loadtxt(f'{Name}.gz').shape)

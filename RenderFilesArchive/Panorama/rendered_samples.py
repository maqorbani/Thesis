# %%
import numpy as np
import matplotlib.pyplot as plt
import os

alt = np.loadtxt('climate_data/Altitude.txt')          # Sun altitude
azi = np.loadtxt('climate_data/Azimuth.txt') - 180     # Sun azimuth
# Hour of year for each key
key = np.loadtxt('climate_data/key.txt', dtype='str')
dered = os.listdir('Octs')

# %%
render = [list(key).index(i)
          for i in dered if os.path.exists(f'Octs/{i}/done')]

# %%
fig, ax1 = plt.subplots(1, figsize=(10, 7))
plt.scatter(azi, alt, s=5)
plt.scatter(azi[render], alt[render], c='red', s=10)

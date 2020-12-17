# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# from IPython import get_ipython
# %%
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

# %%
'''
with open('Altitude.txt') as f:
    alt = f.read()
    alt = alt.split('\n')
    alt = [float(i) for i in alt]

# %%
with open('Azimuth.txt') as f:
    azi = f.read()
    azi = azi.split('\n')
    azi = [float(i) for i in azi]

# %%
with open('dirRad.txt') as f:
    dire = f.read()
    dire = dire.split('\n')
    dire = [float(i) for i in dire]

# %%
with open('difHorRad.txt') as f:
    dif = f.read()
    dif = dif.split('\n')
    dif = [float(i) for i in dif]
# %%

'''
# %%
alt = np.loadtxt('Altitude.txt')
azi = np.loadtxt('Azimuth.txt') - 180
dire = np.loadtxt('dirRad.txt')
dif = np.loadtxt('difHorRad.txt')
key = np.loadtxt('key.txt', dtype='str')
print(len(alt), len(azi), len(dire), len(dif))

# %%
plt.scatter(azi, alt, s=10)
plt.xlabel('azimuth')
plt.ylabel('altitude')
plt.show()

# %%
plt.scatter(dire, dif, s=10)
plt.xlabel('direct radiation')
plt.ylabel('diffuse radiation')
plt.show()

# %%
sunData = np.vstack((alt, azi, dire, dif)).T

# %%
# Run the K-means algorithm to cluster data into k groups.
k = 200
kmeans = KMeans(n_clusters=k, max_iter=9000).fit(sunData)

# %%


def euclidean_distance(points, centroid):
    clusters = kmeans.predict(points)
    distances = {}
    for i in range(k):
        distances[i] = [200, 0]

    # print(distances)

    for i, p in enumerate(points):
        p = p.reshape(1, 4)
        dist = np.linalg.norm((p - centroid[clusters[i]]))
        # print(distances[i])
        if dist < distances[clusters[i]][0]:
            distances[clusters[i]][0] = dist
            distances[clusters[i]][1] = i

    # represent = {}
    # for i, value in enumerate(distances):
    #     gg
    return distances


distances = euclidean_distance(sunData, kmeans.cluster_centers_)
# %%
selected = []
selectKeys = []

for i in distances.values():
    selected.append([alt[i[1]], azi[i[1]], dire[i[1]], dif[i[1]]])
    selectKeys.append(i[1])

selected = np.array(selected)
selectKeys = np.array(selectKeys)

# %%
plt.scatter(azi, alt, s=2, c='grey')
plt.scatter(selected[:, 1], selected[:, 0], s=10, c='red')
plt.xlabel('azimuth')
plt.ylabel('altitude')
plt.show()

# %%
plt.scatter(dire, dif, s=2, c='grey')
plt.scatter(selected[:, 2], selected[:, 3], s=10, c='red')
plt.xlabel('direct rad')
plt.ylabel('diffuse rad')
plt.show()

# %%
for i in distances:
    print(key[distances[i][1]])

# FIND OUT IF THERE ARE ANY coincidental CENTROIDS
# %%
coincidental = np.zeros((k, k))
center = kmeans.cluster_centers_

for i in range(k):
    coincidental[:, i] = np.linalg.norm((center-center[i]), axis=1)

(coincidental+np.eye(200)*coincidental.max()).min(axis=0)  # Gets dist matrix

# %%
np.savetxt('results.txt', selectKeys, fmt='%s')

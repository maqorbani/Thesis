# %%
import os

with open('skyResearch/skiesToCreate.txt', 'r') as f:
    skies = f.read()

with open('skyResearch/climateBasedSky@_1_1@900.sky', 'r') as f:
    baseSky = f.readlines()

with open('skyResearch/key.txt', 'r') as f:
    skyKey = f.read()

with open('skyResearch/dirRad.txt', 'r') as f:
    dirRad = f.read()

with open('skyResearch/difHorRad.txt', 'r') as f:
    difHorRad = f.read()

with open('skyResearch/Altitude.txt', 'r') as f:
    Altitude = f.read()

with open('skyResearch/Azimuth.txt', 'r') as f:
    Azimuth = f.read()

skies = skies.split(',')
skyKey = skyKey.split('\n')
dirRad = dirRad.split('\n')
difHorRad = difHorRad.split('\n')
Altitude = Altitude.split('\n')
Azimuth = Azimuth.split('\n')
# print(baseSky[2].replace('altitude', '20'))
indices = ([skyKey.index(skies[i]) for i, _ in enumerate(skies)])

createdSky = {}
for i in skies:
    j = skyKey.index(i)
    createdSky[j] = baseSky.copy()
    createdSky[j][2] = createdSky[j][2].replace('altitude', Altitude[j])
    createdSky[j][2] = createdSky[j][2].replace('azimuth', Azimuth[j])
    createdSky[j][2] = createdSky[j][2].replace('direct', dirRad[j])
    createdSky[j][2] = createdSky[j][2].replace('diffuse', difHorRad[j])

# %%
for i in createdSky:
    with open(f'skyResearch/climateBasedSky@_{skyKey[i]}.sky', 'w') as f:
        f.write(''.join(createdSky[3367]))
        print(i)

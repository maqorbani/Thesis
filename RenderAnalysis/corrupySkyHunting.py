import os

with open('badDirs.txt') as f:
    badDirs = f.read()

with open('key.txt') as f:
    keys = f.read()

badDirs = badDirs.split(',')[:-1]
# print(badDirs)
keys = keys.split('\n')

skies = os.listdir('TheRender/Tehran_Mehrabad_IRN/')
skies = [i[17:-4] for i in skies]

rmIt = []
for i in keys:
    i not in skies and rmIt.append(i)

print(rmIt)

import os

os.chdir('Tehran_Mehrabad_IRN')
skies = []

for i in os.listdir():
	with open(i, 'r') as f:
		sky = f.read()
	
	if '-W 0.0 0.0' in sky:
		skies.append(i)

for i in skies:
	print(i)
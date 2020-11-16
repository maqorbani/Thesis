import os

os.chdir('Octs')
dirs = []

for i in os.listdir():
	if os.path.isdir(i):
		os.chdir(i)
		if len(os.listdir()) < 2:
			dirs.append(i)
		else:
			print(i)
		os.chdir('../')

print(dirs)

import os

os.chdir('Octs')
dirs = []

for i in os.listdir():
	if os.path.isdir(i):
		os.chdir(i)
		if len(os.listdir()) > 2:
			dirs.append(i)
		else:
                        pass
		os.chdir('../')

print(dirs)
print(len(dirs), ' of ', len(os.listdir()))

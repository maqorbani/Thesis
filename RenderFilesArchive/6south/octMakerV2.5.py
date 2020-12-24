import os
import subprocess
import shutil
import concurrent.futures
# import numpy as np

nCPU = 256  # Number of CPUs to split work to

# os.chdir('Desktop/TheRender/')
print('Current working dir is: ' + os.getcwd())
skies = os.listdir('../Tehran_Mehrabad_IRN/')  # 4399
# skies = skies[:3]  # To test a few number

# Omitting rendered skies from the list
dirs = []

for i in os.listdir('Octs/'):
    if os.path.isdir('Octs/'+i):
        if len(os.listdir('Octs/'+i)) != 2:
            dirs.append(i)
            shutil.rmtree('Octs/'+i)
        else:
            try:
                skies.remove('climateBasedSky@_'+i+'.sky')
            except ValueError:
                print('Warning! x not in list')

# print(skies) 
divisionCPU = len(skies) // nCPU

skyDict = {}
for i in range(nCPU):
    skyDict['sky'+str(i)] = skies[i*divisionCPU:divisionCPU*(i+1)]

for i, sky in enumerate(skies[nCPU * divisionCPU:]):
    skyDict['sky'+str(i)].append(sky)
print(skyDict)
# print(len(list(skyDict.values())[5]))

# read the oconv command from file
with open('IMGInit.bat', 'r') as f:
    a = f.read()

a = a.split()

# read the rpict command from file
with open('rpict.bat', 'r') as f:
    render = f.read()
render = render.rstrip("\n")

# read the pfilt command from file
with open('pfilt.bat', 'r') as f:
    pfilt = f.read()

pfilt = pfilt.split()

# ___________>>>
os.chdir('Octs')

# Remove former dirs
# for dir in os.listdir():  
#     if os.path.isdir(dir):
#         shutil.rmtree(dir)

# Set the PATH variables
os.environ['PATH'] += os.pathsep + ':/usr/local/radiance/bin'
os.environ['RAYPATH'] = '.:/usr/local/radiance/lib'


def octMaker(sky, j):
    for i in sky:
        os.mkdir(str(i[17:-4]))
        os.chdir(str(i[17:-4]))
        b = a  # b is some arbitrary variable
        b[4] = "../../../Tehran_Mehrabad_IRN/" + str(i)
        b = ' '.join(b)
        with open(str(i[17:-4]), 'w') as f:  # Write the octree executable
            f.write(b)
        os.system('chmod +x ./*')  # chnage mode to +x
        os.system('./*')  # execute the oconv command
        os.remove(str(i[17:-4]))  # remove the octree execuatable
        
        os.system(render + str(i[17:-4] + '.HDR'))  # renders using rpict
        os.remove('thesisRoom11_5_IMG.oct')  # remove the octree

        # with open('render', 'w') as f:  # Write the render executable
        #     f.write(render + str(i[17:-4] + '.HDR'))
        # os.system('chmod +x render')
        # os.system('render')  # render using rpict

        b = pfilt  # anti aliasing stuff
        b[7] = i[17:-4] + '.HDR'
        b = '  '.join(b)
        b += i[17:-4] + '_1.HDR'
        os.system(b)  # end of anti aliasing stuff

        os.remove(i[17:-4] + '.HDR')

        print(str(i[17:-4] + ' done!'))
        os.chdir('../')  # Get back one level up

    return 'Process #' + str(j) + ' is done!'


with concurrent.futures.ProcessPoolExecutor() as executor:
    results = [executor.submit(octMaker, list(
        skyDict['sky'+str(i)]), i) for i in range(nCPU)]

    for f in concurrent.futures.as_completed(results):
        print(f.result())

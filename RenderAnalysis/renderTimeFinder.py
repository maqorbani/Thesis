import re
import os

pattern = re.compile(r'(\d.\d{3})r')

for i in os.listdir('TheRender/Octs/'):
    with open(f'TheRender/Octs/{i}/error.log') as f:
        rTime = f.readlines()[-1]
    
    print(float(re.findall(pattern, rTime)[0]))

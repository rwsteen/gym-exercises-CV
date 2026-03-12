import numpy as np
import scipy.io
import os
import shutil
import glob
from pathlib import Path

mat_path = os.path.join('Penn_Action','Penn_Action','labels')
base_dir = "augmented_penn"

if os.path.exists(base_dir):
    shutil.rmtree(base_dir)
    
src_tool = os.path.join('Penn_Action','Penn_Action','tools')
tool = os.path.join(base_dir, "tools")
shutil.copytree(src_tool, tool)
os.makedirs(os.path.join(base_dir, "labels"))
os.makedirs(os.path.join(base_dir, "frames"))

for file in os.scandir(mat_path):
    if Path(file).suffix == '.mat':
        mat = scipy.io.loadmat(file)
        #print(mat['action'])
        if mat['action'] == ['pushup'] or mat['action'] == ['squat'] or mat['action'] == ['pullup'] or mat['action'] == ['bench_press'] or mat['action'] == ['situp']:
            src = os.path.join(mat_path,file.name)
            lbl = os.path.join(base_dir, "labels", file.name)
            folder_name = os.path.splitext(file.name)[0]
            src_frame = os.path.join('Penn_Action','Penn_Action','frames', folder_name)
            frame = os.path.join(base_dir, "frames", folder_name)
            if os.path.isfile(src):
                shutil.copy2(src, lbl)
                shutil.copytree(src_frame, frame)

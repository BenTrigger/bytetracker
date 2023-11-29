import shutil
import os
import time
starttime = time.time()
src = r'/workspace/checkspeed/checkspeed'
dest = r'/MyHomeDir/checkspeed/results'
src_files = os.listdir(src)
for file_name in src_files:
    fu_file_name = os.path.join(src, file_name)
    if os.path.isfile(fu_file_name):
        shutil.copy(fu_file_name, dest)
endtime = time.time()
print(endtime - starttime)

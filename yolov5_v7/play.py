import pandas as pd
import glob
import os

src = '/home/user1/ariel/GNNs/101m.txt'
pdb = []
with open(src, 'r') as f:
    lines = f.readlines()
for line in lines:
    if line.startswith('ATOM'):
        pdb.append(line)
    elif line.startswith('HETATM'):
        pdb.append(line)
col_names = ['ATOM','ind','at_in_pep','aa','chain','aa_ind','x','y','z','unknown1','unknown2','at_type']
df = pd.DataFrame()
dict_list = []
for line in pdb:
    new_line = line.split(' ')
    res = []
    for ele in new_line:
        if ele.strip():
            res.append(ele)
    dic = {k: v for k, v in zip(col_names, res)}
    dict_list.append(dic)
    #data = pd.DataFrame([res], columns=col_names)

df = pd.DataFrame.from_dict(dict_list)
print('g')
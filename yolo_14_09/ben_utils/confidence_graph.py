from pathlib import Path
from glob import glob
from tqdm import tqdm
dict_map = {}
import matplotlib.pyplot as plt
import seaborn as sns
names = ['Person', 'Person on board a vessel', 'Swimmer', 'Sail boat', 'Floating object',  'Dvora', 'Zeara', 'PWC', 'Merchant Ship', 'Inflatable Boat', 'Vessel']
save_dir = r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\runs\compare_zafrir\confidense_graph\plot_of_type_'
files_path = glob(r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\runs\val\val_on_ashdod_weights_ashdod_erez_with_conf\labels\*.txt') #input real lbls
print(len(files_path))
for file in tqdm(files_path, total=len(files_path)):
    with open(file, encoding="utf-8") as f:
        try:
            lines = f.readlines()
        except Exception as e:
            print(Path(file).stem)
        for line in lines:
            values_in_line = line.split(' ')
            if values_in_line[0] in dict_map:
                dict_map[values_in_line[0]].append(float(values_in_line[5]))  #dict[type of target] = conf
            else:
                dict_map[values_in_line[0]] = [float(values_in_line[5])]
keys = dict_map.keys()
for key in keys:
    values = dict_map[key] # values of type
    sns.displot(values, bins=40).set(title=names[int(key)])
    plt.savefig(save_dir + names[int(key)]+"_"+key)


import numpy
from tqdm import tqdm
from glob import glob
from matplotlib import Path
import numpy as np

prediction_labels_folder = r'Z:\deepsort_yolov5\deepsort_ben\runs\track\mul_4_custom_weights_all_data_deep_crop_226\labels\*'
#prediction_labels_folder = r'/MyHomeDir/deepsort_yolov5/deepsort_ben/runs/track/mul_4_custom_weights_all_data_deep_crop_226/labels/*'
path_to_preds = glob(prediction_labels_folder)
width = 3840
height = 2160
print(len(path_to_preds))

arr_res = np.array([])

for file in tqdm(path_to_preds):
    output_folder = Path(file).parents[1] / 'object_sizes'
    output_folder.mkdir(parents=True, exist_ok=True)
    if not Path(file).exists():
        print('Empty File: %s' % file)
    with open(file, 'r', encoding='UTF-8') as f_r:
        x = np.loadtxt(f_r, delimiter=' ', dtype=np.float32)
        x = x[x[:, 3] > 0]
        y = x.copy()

        y[:,1] = x[:, 3] * width
        y[:,2] = x[:, 4] * height

        # file_to_write  = str(output_folder / Path(file).name)
        # with open(file_to_write,'wb') as f_w:
        #     np.savetxt(f_w, y[:,0:3], fmt='%.2f')

        if arr_res.size == 0:
            arr_res = y[:,0:3]
        else:
            arr_res = numpy.vstack([arr_res,y[:,0:3]])

file_all_in_one = r'Z:\deepsort_yolov5\deepsort_ben\runs\track\mul_4_custom_weights_all_data_deep_crop_226\object_sizes\all_results_no_zeros.txt'
with open(file_all_in_one,'wb') as f_w:
    np.savetxt(f_w, arr_res, fmt='%.2f')

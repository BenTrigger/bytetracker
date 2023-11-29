
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from utils.metrics import ConfusionMatrix , ap_per_class
from glob import glob
from utils.general import xywh2xyxy
import numpy as np
import torch
from tqdm import tqdm
from matplotlib import Path
from val import process_batch
# Arguments:
#             detections (Array[N, 6]), x1, y1, x2, y2, conf, class
#             labels (Array[M, 5]), class, x1, y1, x2, y2
#             false_save_class int


def txt2tensor(file, pred=False):
    if not Path(file).exists():
        return None

    file = open(file, 'r', encoding='UTF-8')
    x = np.loadtxt(file, delimiter = ' ', dtype=np.float32)
    if len(x.shape) == 1:
        x = x.reshape(1,-1)
    y = x.copy()
    if pred:
        y[:,-1] = x[:,0] # class to last position
        #IF NO CONF
        #y = np.insert(y,4,1.0,axis=1) # add score 1.0
        #ELSE
        y[:,-2] = x[:,5] # y[ CONF ]  = X[: CONF]
        y[:,:4] = xywh2xyxy(x[:,1:5]) #x1,y1,x2,y1 at the beginning
    else:
        y[:,1:] = xywh2xyxy(x[:,1:])
    tens = torch.tensor(y)
    return tens


#LOCAL RUN
out_dir = r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\check_res\Deep_Sort'
#out_dir = r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\check_res\Yolo'
imageOutputPath = r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\check_res\Deep_Sort\image_outpath'
if not ((os.path.exists(out_dir)) or (os.path.exists(imageOutputPath))):
    os.mkdir(out_dir), os.mkdir(imageOutputPath)

images_folder = r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\data\atr_zafrir\tagged_14_02_22\images\val\*'
prediction_labels_folder = r'\\27.30.3.26\uxcag_users\u30111\deepsort_yolov5\Yolov5_DeepSort_Pytorch-master\runs\track\val_set_deep_custom_weights7\labels\*'
#prediction_labels_folder = r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\runs\val\val_on_ashdod_weights_ashdod_erez_with_conf\labels\*'
true_labels_folder = r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\data\atr_zafrir\tagged_14_02_22\labels\val\*'

# SERVER RUN
# out_dir = r'/MyHomeDir/yolo_14_09/check_res/results'
# imageOutputPath = r'/MyHomeDir/yolo_14_09/check_res/results/image_outpath'
# if not ((os.path.exists(out_dir)) or (os.path.exists(imageOutputPath))):
#     os.mkdir(out_dir), os.mkdir(imageOutputPath)
#
# images_folder = r'/MyHomeDir/yolo_14_09/data/atr_zafrir/tagged_14_02_22/images/val/*'
# prediction_labels_folder = r'/MyHomeDir/deepsort_yolov5/Yolov5_DeepSort_Pytorch-master/runs/track/val_set_deep_custom_weights7/labels/*'
# #prediction_labels_folder = r'/MyHomeDir/yolo_14_09/runs/val/val_on_ashdod_weights_ashdod_erez_with_conf/labels/*'
# true_labels_folder = r'/MyHomeDir/yolo_14_09/data/atr_zafrir/tagged_14_02_22/labels/val/*'

nc = 11
conf_thres = torch.tensor([0.24] * nc)
confusion_iou_thres = 0.002
false_save_class = [0,1,2,3,4,5,6,7,8,9,10]
names = {0: 'Person', 1: 'Person on board a vessel', 2: 'Swimmer', 3: 'Sail boat', 4: 'Floating object', 5: 'Dvora', 6: 'Zeara', 7: 'PWC', 8: 'Merchant Ship', 9: 'Inflatable Boat', 10: 'Vessel'}
#names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
confusion_matrix = ConfusionMatrix(nc=nc, conf=conf_thres, iou_thres=confusion_iou_thres)
path_images = sorted(glob(images_folder))
path_to_preds = sorted(glob(prediction_labels_folder))
path_to_reals = sorted(glob(true_labels_folder))

# for img_path in path_images:
#     lbl_in_pred = prediction_labels_folder + str(Path(str(Path(img_path).stem)  + '.txt'))
#     print(lbl_in_pred)
#     if not Path(lbl_in_pred).exists():
#         pass
#         #create an empty label file.txt
seen = 0
device = 0 # 'cpu'
iouv = torch.linspace(0.5, 0.95, 10).to('cpu')  # iou vector for mAP@0.5:0.95
niou = iouv.numel()
stats = []

for img_path in tqdm(path_images):
    #TODO go through the images folder and if no real label but prediction add to FP
    val_real = true_labels_folder[:-1] + str(Path(img_path).stem + '.txt')
    lbl_in_pred = prediction_labels_folder[:-1] + str(Path(val_real).name)

    predn = txt2tensor(lbl_in_pred, True)
    labelsn = txt2tensor(val_real)
    if predn is None and labelsn is None:
        continue

    if labelsn is not None and predn is None: # only labels and no prediction
        nl = len(labelsn)
        tcls = labelsn[:, 0].tolist() if nl else []
        appen_for_stats = (torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls)
    elif labelsn is None and predn is not None: # only prediction and not labels
        correct = torch.zeros(predn.shape[0], niou, dtype=torch.bool)
        appen_for_stats =(correct.cpu(), predn[:, 4].cpu(), predn[:, 5].cpu(), tcls) # (correct, conf, pcls, tcls)
    else: # have predictions and labels
        nl = len(labelsn)
        tcls = labelsn[:, 0].tolist() if nl else []
        correct = process_batch(predn, labelsn, iouv) #TODO need to make it happen
        appen_for_stats =(correct.cpu(), predn[:, 4].cpu(), predn[:, 5].cpu(), tcls) # (correct, conf, pcls, tcls)

    confusion_matrix.process_batch(predn, labelsn, false_save_class)
    seen += 1
    stats.append(appen_for_stats)
try:
    confusion_matrix.plot(save_dir=out_dir, names=list(names.values()))
except Exception as e:
    print(e)

# Compute statistics
stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
if len(stats) and stats[0].any():
    p, r, ap, f1, ap_class = ap_per_class(*stats, plot=True, save_dir=out_dir, names=names)
    ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
else:
    nt = torch.zeros(1)

# Print results
pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

for i, c in enumerate(ap_class):
    print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

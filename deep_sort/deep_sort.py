import numpy as np
import torch
import sys
import gdown
from os.path import exists as file_exists, join
from pathlib import Path
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.detection import Detection
from .sort.tracker import Tracker
from .deep.reid_model_factory import show_downloadeable_models, get_model_link, is_model_in_factory, \
    is_model_type_in_model_path, get_model_type, show_supported_models
import torchvision.models as models
import torchvision.transforms as T
sys.path.append('deep_sort/deep/reid')

# from torchreid.utils import FeatureExtractor
# from torchreid.utils.tools import download_url
from torchreid import utils as torchreidutils

from deep_sort.networks.resnet_big import SupConResNet
import torch.nn as nn
show_downloadeable_models()

__all__ = ['DeepSort']

class CustomFeatureExtractor(torch.nn.Module):
    def __init__(self, model_name, model_path, device):
        super().__init__()
        self.model_name = model_name
        self.device = device
        if model_name == 'resnet50':
            model = set_supCon
        else:
            model = set_benNet
        self.features, mean, std = model(model_name, model_path, device)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize(128),
            #T.Resize(226),
            #T.Resize(32),
            T.CenterCrop(128),
            #T.CenterCrop(224),
            #T.CenterCrop(32),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])

    def forward(self, xs):
        x_list = []
        for x in xs:
            x_list.append(self.transform(x))

        x_tensor = torch.stack(x_list).to(self.device)
        y = self.features(x_tensor)
        return y

def set_supCon(model, path_to_model,device):
    model = SupConResNet(name=model)
    model= model.to(device)
    #model = model
    print(path_to_model)
    ckpt = torch.load(path_to_model, map_location=device)
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model
        model.load_state_dict(state_dict)
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2675, 0.2565, 0.2761]
    return model, mean, std

def set_benNet(model, path_to_model, device):
    model = torch.load(path_to_model, map_location=device)
    features = torch.nn.ModuleList(model.children())[:-1]
    features = torch.nn.Sequential(*features)
    mean = [0.485, 0.456, 0.406] # RESNET50
    std = [0.229, 0.224, 0.225] # RESNET50
    return features, mean, std


class DeepSort(object):
    def __init__(self, model, device, max_dist=0.2, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100):
        if is_model_in_factory(model):
            print("in factory")
            # download the model
            model_path = join('deep_sort/deep/checkpoint', model + '.pth')
            if not Path(model_path).exists():
                model_path = join('deep_sort/deep/checkpoint', model + '.pt')
                if not Path(model_path).exists():
                    gdown.download(get_model_link(model), model_path, quiet=False)
                else:
                    print('found')
            else:
                print('found')
            try:
                nam = model.rsplit('_', 1)[:-1][0] if 'b7' not in model else model
                print(nam)
            except:
                nam = model

            if nam in ['custom_data_ckpt_epoch','ckpt_crop128_train']:
                self.extractor = CustomFeatureExtractor('resnet50', model_path, device=str(device))
            elif nam in ['resnet','efficientnet_b7']:
                self.extractor = CustomFeatureExtractor(nam, model_path, device=str(device))
            else:
                module =  torchreidutils.FeatureExtractor
                self.extractor = module(
                    # get rid of dataset information DeepSort model name
                    model_name=nam,#model,
                    model_path=model_path,
                    device=str(device)
                )
        else:
            if is_model_type_in_model_path(model):
                model_name = get_model_type(model)
                print(f'MODEL NAME: {model_name}')
                self.extractor = torchreidutils.FeatureExtractor(
                    # get rid of dataset information DeepSort model name
                    model_name=model_name,
                    model_path=model,
                    device=str(device)
                )
            else:
                print('Cannot infere model name from provided DeepSort path, should be one of the following:')
                show_supported_models()
                exit()

        self.max_dist = max_dist
        print (f'max_dist: {max_dist}')
        metric = NearestNeighborDistanceMetric(
            "cosine", self.max_dist, nn_budget)
        self.tracker = Tracker(
            metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def update(self, bbox_xywh, confidences, classes, ori_img):
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        features = self._get_features(bbox_xywh, ori_img)
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(
            confidences)]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections, classes, confidences)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            
            track_id = track.track_id
            class_id = track.class_id
            conf = track.conf
            outputs.append(np.array([x1, y1, x2, y2, track_id, class_id, conf]))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), self.height - 1)
        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features

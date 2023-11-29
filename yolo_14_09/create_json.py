import json
import cv2 as cv
import os

coco_json = {'info': {'description': 'singapore maritime',
                      'url': '',
                      'version': '1.0',
                      'year': 2021,
                      'contributor': '',
                      'date_created': '2021/03/17'}
             ,
             'licenses': [{'url': '',
                           'id': 1}]
             ,
             'categories': [{'supercategory': 'water vehicle',
                             'id': 1,
                             'name': 'boat'}]
             ,
             'images': []
             ,
             'annotations': []
             }

path = r'\\uxcagstg.maman.iai\\uxcag_users\\u34946\\video_frames_merchant\\'
out_path  = r'\\uxcagstg.maman.iai\\uxcag_users\\u34946\\video_frames_merchant\\'

#path = r'\\uxcagstg.maman.iai\\uxcag_users\\u34946\\tmp\\'
#out_path  = r'\\uxcagstg.maman.iai\\uxcag_users\\u34946\\tmp\\'


for directoy in os.listdir(path):
    id = 0
    images = []
    for f in [g for g in os.listdir(path+directoy) if g[-3:] == 'png']:
        img = cv.imread(path + '//'+directoy + '//'+ f)
        image = {}
        image["height"] = img.shape[0]
        image["width"] = img.shape[1]
        image["id"] = id
        image["file_name"] = f
        images.append(image)
        id=id+1
    coco_json["images"] = images
    json.dump(coco_json, open(f'{out_path+directoy}\coco_json.json', "w"), indent=4)





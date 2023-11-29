import os
#from shapely.geometry import Polygon
import argparse
import numpy as np
import pandas as pd
import cv2
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

def create_slices(image_height: int,
                  image_width: int,
                  overlap_height_ratio=0.2,
                  overlap_width_ratio=0.2,
                  slice_size=1024):
    if isinstance(slice_size, tuple):
        slice_size_x, slice_size_y = slice_size
    else:
        slice_size_x = slice_size
        slice_size_y = slice_size
    num_x_slices = np.ceil(image_width*(1+overlap_width_ratio) / slice_size_x) #3
    num_y_slices = np.ceil(image_height*(1+overlap_height_ratio) / slice_size_y) #2


    x_step = int((image_width - slice_size_x)/(num_x_slices-1)) #1120
    y_step = int((image_height - slice_size_y) / (num_y_slices - 1)) #560

    bboxes = []
    x = 0

    while x+slice_size_x <= image_width:
        y = 0
        while y+slice_size_y <= image_height:
            bboxes.append((x, y, x+slice_size_x, y+slice_size_y))
            y = y + y_step
        x = x + x_step
    # print(bboxes)
    return bboxes


def sliced_tagging(box, tags_pols, size, filename, sliced_size):
    sliced_tag = []
    x1,y1,x2,y2 = box
    #    print('herer2')
    img_pol = Polygon([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])

    for tag in tags_pols:
        # FOR TESTING RESULTS.... BEN
        o,p= tag[1].exterior.coords.xy
        if o[0]==2514:
            print('is intersects? -> %s' % img_pol.intersects(tag[1]))
            print('box is: %s %s %s %s' % box)
            print('tag coords: %s' % tag[1])
        if img_pol.intersects(tag[1]):
            inter = img_pol.intersection(tag[1]) # intersaction polygon
            new_box = inter.envelope # the intersection bbox
            x,y = new_box.exterior.coords.xy
            new_x1 = int(min(x) - x1)
            new_y1 = int(min(y) - y1)
            new_x2 = int(max(x) - x1)
            new_y2 = int(max(y) - y1)
            sliced_tag.append([tag[0], new_x1, new_y1, new_x2, new_y2])

    # write the new tags
    p = Path(filename).parent
    os.makedirs(p, exist_ok=True)
    if os.path.isfile(filename):
        os.remove(filename)
    if len(sliced_tag):
        with open(filename, 'a') as myfile:
            for tag in sliced_tag:
                ybox = bbox_to_yolo(tag, sliced_size)
                mystring = str(
                    str(tag[0]) + " " + str(truncate(float(ybox[1]), 7)) + " " + str(truncate(float(ybox[2]), 7)) + " "
                    + str(truncate(float(ybox[3]), 7)) + " " + str(truncate(float(ybox[4]), 7)))
                myfile.write(mystring)
                myfile.write("\n")
        myfile.close()
    return sliced_tag




def write_sliced_tagging(starting_pixel, all_tags, size, filename, sliced_size):
    slice_tags = []
    sliced_size_x, sliced_size_y = sliced_size
    s_xmin, s_ymin, s_xmax, s_ymax = (starting_pixel[0], starting_pixel[1],
                                      starting_pixel[0] + sliced_size_x, starting_pixel[1] + sliced_size_y)

    for tag in all_tags:  # run over all bbxs
        label, t_xmin, t_ymin, t_xmax, t_ymax = tag
        if (t_xmin<=s_xmax) and (t_xmin>=s_xmin):
            xmin = t_xmin
        else:
            xmin = False

        if (t_xmax<=s_xmax) and (t_xmax>=s_xmin):
            xmax = t_xmax
        else:
            xmax = False
        if (t_ymin<=s_ymax) and (t_ymin>=s_ymin):
            ymin = t_ymin
        else:
            ymin = False
        if (t_ymax<=s_ymax) and (t_ymax>=s_ymin):
            ymax = t_ymax
        else:
            ymax = False

        # all tags in slice
        if xmin and ymin and xmax and ymax:
            # print(1)
            # print(xmin, ymin, xmax, ymax)
            # print(s_xmin, s_ymin, s_xmax, s_ymax)
            pred = [xmin-s_xmin, ymin-s_ymin, xmax-s_xmin, ymax-s_ymin]
        # only left down in slice
        elif xmin and ymax and not ymin and not xmax:
            # print(2)
            pred = [xmin-s_xmin, 0, sliced_size_x, ymax-s_ymin]

        # if only left top in slice
        elif xmin and ymin and not xmax and not ymax:
            # print(3)
            pred = [xmin-s_xmin, ymin-s_ymin, sliced_size_x, sliced_size_y]

        # if left top and down in slice
        elif not xmax and xmin and ymin and ymax:
            # print(4)
            pred = [xmin-s_xmin, ymin-s_ymin, sliced_size_x, ymax-s_ymin]

        # if right and left top in slice
        elif not ymax and xmin and ymin and xmax:
            # print(5)
            pred = [xmin-s_xmin, ymin-s_ymin, xmax-s_xmin, sliced_size_y]

        # if only right top in slice
        elif not ymax and not xmin and ymin and xmax:
            # print(6)
            pred = [0, ymin - s_ymin, xmax - s_xmin, sliced_size_y]

        # if bottom right and top right in slice
        elif not xmin and ymax and ymin and xmax:
            # print(7)
            pred = [0, ymin-s_ymin, xmax-s_xmin, ymax-s_ymin]

        # if only bottom right in slice
        elif not xmin and not ymin and xmax and ymax:
            # print(8)
            pred = [0, 0, xmax-s_xmin, ymax-s_ymin]

        # if right and left bottom in slice
        elif not ymin and xmin and xmax and ymax:
            # print(9)
            pred = [xmin-s_xmin, 0, xmax-s_xmin, ymax-s_ymin]

        else:
            # print(10)
            pred = [0, 0, 0, 0]

        area = (pred[2]-pred[0])*(pred[3]-pred[1])

        if area > 0.00000001: # BEN FIX, WAS 60
            slice_tags.append([int(label)] + pred)
    p = Path(filename).parent

    os.makedirs(p, exist_ok=True)
    if os.path.isfile(filename):
        os.remove(filename)
    if len(slice_tags):
        with open(filename, 'a') as myfile:
            for box in slice_tags:
                ybox = bbox_to_yolo(box, sliced_size)
                # ybox = box
                mystring = str(
                    str(ybox[0]) + " " + str(truncate(float(ybox[1]), 7)) + " " + str(truncate(float(ybox[2]), 7)) + " "
                    + str(truncate(float(ybox[3]), 7)) + " " + str(truncate(float(ybox[4]), 7)))
                myfile.write(mystring)
                myfile.write("\n")
        myfile.close()
    return slice_tags


def bbox_to_yolo(bbox, size):
    if isinstance(size, tuple):
        width , height = size
    else:
        width = size
        height = size
    dw = 1. / width
    dh = 1. / height

    label = bbox[0]
    xmin = bbox[1]
    ymin = bbox[2]
    xmax = bbox[3]
    ymax = bbox[4]

    x = (xmin + xmax) / 2
    y = (ymin + ymax) / 2

    w = xmax - xmin
    h = ymax - ymin

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh

    return [label, x, y, w, h]

def yolo_to_norm_not_file(bbox, size):
    dw, dh = size
    label, x, y, w, h = bbox
    l = int((x - w / 2) * dw)
    r = int((x + w / 2) * dw)
    t = int((y - h / 2) * dh)
    b = int((y + h / 2) * dh)
    if l < 0:
        l = 0
    if r > dw - 1:
        r = dw - 1
    if t < 0:
        t = 0
    if b > dh - 1:
        b = dh - 1
    x_min, y_min = l, t
    x_max, y_max = r, b
    return [label, x_min, y_min, x_max, y_max]

def yolo_to_norm(tag_file, size):
    """
    transforms yolo type labels into normal pixel labels
    Parameters
    ----------
    tag_file: path to the label txt
    size: origin size of corresponding image (width, height)

    Returns
    -------
    list of bboxes converted to normal coordinates (x_min, y_min, x_max, y_max)

    """
    dw, dh = size
    if os.path.getsize(tag_file):
        data = pd.read_csv(tag_file, header=None, sep=" ").to_numpy()
    else:
        return []
    pixel_coords = []
    try:
        for dt in data:
            label, x, y, w, h = dt
            l = int((x - w / 2) * dw)
            r = int((x + w / 2) * dw)
            t = int((y - h / 2) * dh)
            b = int((y + h / 2) * dh)
            if l < 0:
                l = 0
            if r > dw - 1:
                r = dw - 1
            if t < 0:
                t = 0
            if b > dh - 1:
                b = dh - 1
            x_min, y_min = l, t
            x_max, y_max = r, b
            pixel_coords.append([label, x_min, y_min, x_max, y_max])
    except Exception as e:
        print('label with problem check it %s' % tag_file)
    return pixel_coords


def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier


def cut_tag_and_img(image_path, tag_path, output_path_images, output_path_labels, save_pred_pics, overlap_pct=0.1, size=1024, boxes=None, with_background=False):
    """
    takes an image and its label and slices both to wanted size
    Parameters
    ----------
    image_path
    tag_path
    out_path
    save_pred_pics
    overlap_pct
    size

    Returns
    -------

    """
    assert boxes != None
    image_array = cv2.imread(image_path)
    try:
        height, width, _ = image_array.shape
    except Exception as e:
        print("could not load img path: %s" % image_path)
        return
    image_name = Path(image_path).stem
    try:
        all_tags = yolo_to_norm(tag_path, (width, height))
    except:
        all_tags = []
    tags_pols = []
    #all tags in the whole image into p
    for tag in all_tags:
        label,x1,y1,x2,y2 = tag
        #tags_pols.append((int(label), Polygon([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])))
        # if int(label) == 10:
        #     print([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])
    colors = plt.cm.get_cmap('Paired').colors
    os.makedirs(output_path_images, exist_ok=True)
    os.makedirs(output_path_labels, exist_ok=True)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        sliced_array = image_array[y1:y2, x1:x2]
        #sliced_tag_path = f'{output_path_labels}/{image_name}_{i}.txt'
        sliced_tag_path = str(Path(output_path_labels) / f'{image_name}_{i}.txt')
        tags = write_sliced_tagging(starting_pixel=(x1, y1), all_tags=all_tags, size=(width, height),
                                    filename=sliced_tag_path, sliced_size=size)
        # print("OLD TAGS:")
        # print(tags)
        # print("NEW TAGS:")
        ##### new slice
        #tags = sliced_tagging(box, tags_pols=tags_pols, size=(width, height),
        #                            filename=sliced_tag_path, sliced_size=size)
        #print(tags)
        ##end new slice
        if tags or with_background:
            cv2.imwrite(f'{output_path_images}/{image_name}_{i}.png', sliced_array)
            if save_pred_pics:
                pred_slice = np.array(sliced_array)
                os.makedirs(f'{output_path_images}/sliced_preds/', exist_ok=True)
                for k, tag in enumerate(tags):
                    r, g, b = colors[tag[0]]
                    cv2.rectangle(
                        pred_slice,
                        tuple(tag[1:3]),
                        tuple(tag[3:5]),
                        color=(int(r * 255), int(g *255), int(b*255)),
                        thickness=2,
                    )
                cv2.imwrite(f'{output_path_images}/sliced_preds/{image_name}_{i}.png', pred_slice)


def run_slicer(path_to_images, path_to_tags, output_path_images, output_path_labels, save_pred_pics, overlap_pct, size, img_w, img_h, with_background=False):
    print(path_to_images)
    formats = ['jpg', 'JPG', 'png', 'PNG']
    boxes = create_slices(img_h, img_w, overlap_pct, overlap_pct, size) # CHECK THIS FUNCTION
    if not os.path.isdir(path_to_images):
        cut_tag_and_img(path_to_images, path_to_tags, output_path_images, output_path_labels, save_pred_pics, overlap_pct, size=size, boxes=boxes, with_background=with_background)

    else:
        image_files = []
        for ext in formats:
            image_files.extend(glob.glob(f'{path_to_images}/*.{ext}'))
            # print(image_files[491])
            # exit(1)
        for image in tqdm(image_files, total=len(image_files)):
            name = Path(image).stem
            tag_path = f'{path_to_tags}/{name}.txt'
            if not os.path.isfile(tag_path) and not with_background:
                with open(tag_path, 'a') as tf:
                    pass
                tf.close()
            cut_tag_and_img(image, tag_path, output_path_images, output_path_labels, save_pred_pics, overlap_pct, size=size, boxes=boxes, with_background=with_background)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=r'/MyHomeDir/yolo_14_09/data/Baloons_DATA/tagged_2_2_2023_all_frames/sliced_1600_no_bird_no_light_drone')
    #parser.add_argument('--data', type=str, default=r'Z:\yolo_14_09\data\Baloons_DATA\tagged_2_2_2023_all_frames') #LOCAL
    # r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\data\atr_zafrir\tagged_23_12_2021')
    #parser.add_argument('--data', type=str, default=r'\\27.30.3.26\uxcag_users\u30111\yolo_14_09\data\atr_zafrir\tagged_23_12_2021')
    #parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=1600, help='train, val image size (pixels)')
    #parser.add_argument('--imgsz', '--img', '--img-size', type=tuple, default=(1280, 720), help='train, val image size (pixels)') # SLICE SICE 640 OR 1600 # default=0.2 SHAY
    parser.add_argument('--imgsz', '--img', '--img-size', type=tuple, default=(1600, 1600), help='train, val image size (pixels)') # SLICE SICE 640 OR 1600 # default=0.2 SHAY
    parser.add_argument('--img_w',  type=int, default=3840, help='dataset image width')
    parser.add_argument('--img_h',  type=int, default=2160, help='dataset image height')
    parser.add_argument('--save_pred_pics', action='store_true', default=False, help='')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--overlap', type=float, default=0.2) # default=0.1 SHAY
    opt = parser.parse_args()
    with_background = False # BEN ADD BACKGROUND WITH SLICING
    target_folder = 'test'  #  'train'   /   'val'    /   'test'
    # images = opt.data + '/images/' + target_folder #+ '/black_panther_04-06-2021_19_20-19_25_cam_1_frame_1465.jpg' # delete /black...  BEN
    # tag_file = opt.data + '/labels/' + target_folder #+ '/black_panther_04-06-2021_19_20-19_25_cam_1_frame_1465.txt' # delete /black... BEN
    # out_path_images = opt.data + '/sliced_1600/' + '/images/' + target_folder
    # out_path_labels = opt.data + '/sliced_1600/' + '/labels/' + target_folder
    # images = str(Path(opt.data) / 'images' / target_folder)
    # tag_file = str(Path(opt.data) / 'labels' / target_folder)
    images = str(Path(opt.data) / 'images' / target_folder)
    tag_file = str(Path(opt.data) / 'labels' / target_folder)
    out_path_images = str(Path(opt.data) / 'sliced' / 'images' / Path(target_folder)) #+ '_with_background'))  #/ Path(target_folder))  ,  / 'sliced_1280_720' /
    out_path_labels = str(Path(opt.data) / 'sliced' / 'labels' / Path(target_folder)) #+ '_with_background'))  #/ Path(target_folder))  ,  / 'sliced_1280_720' /
    print(opt)
    #opt.save_pred_pics = False
    run_slicer(images, tag_file, out_path_images, out_path_labels, save_pred_pics=opt.save_pred_pics, overlap_pct=opt.overlap, size=opt.imgsz, img_w=opt.img_w, img_h=opt.img_h, with_background=with_background)

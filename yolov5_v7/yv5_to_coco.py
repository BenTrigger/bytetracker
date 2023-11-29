import os
import json
from collections import defaultdict
from PIL import Image

def parse_yolov5_label(label_path, image_width, image_height):
    with open(label_path, 'r') as file:
        lines = file.readlines()

    annotations = []
    for line in lines:
        class_id, x_center, y_center, width, height = map(float, line.strip().split())
        x_min = int((x_center - width / 2) * image_width)
        y_min = int((y_center - height / 2) * image_height)
        x_max = int((x_center + width / 2) * image_width)
        y_max = int((y_center + height / 2) * image_height)

        annotation = {
            "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
            "category_id": int(class_id),
            "iscrowd": 0,
            "area": (x_max - x_min) * (y_max - y_min)
        }
        annotations.append(annotation)

    return annotations

def yolov5_to_coco(yolov5_labels_folder, output_json_path, image_folder):
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [],
    }

    categories = defaultdict(lambda: len(categories))
    images = os.listdir(image_folder)
    image_id = 0
    annotation_id = 0

    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        image = {
            "file_name": image_name,
            "height": 0,
            "width": 0,
            "id": image_id,
        }

        image_data = Image.open(image_path)
        image["height"] = image_data.height
        image["width"] = image_data.width

        label_path = os.path.join(yolov5_labels_folder, os.path.splitext(image_name)[0] + '.txt')
        annotations = parse_yolov5_label(label_path, image["width"], image["height"])
        for annotation in annotations:
            annotation["image_id"] = image_id
            annotation["id"] = annotation_id
            annotation_id += 1
            coco_data["annotations"].append(annotation)

        coco_data["images"].append(image)
        image_id += 1

    for category_name, category_id in categories.items():
        category = {
            "supercategory": "object",
            "id": category_id,
            "name": category_name,
        }
        coco_data["categories"].append(category)

    with open(output_json_path, 'w') as output_file:
        json.dump(coco_data, output_file, indent=4)

if __name__ == "__main__":
    yolov5_labels_path = "/home/user1/ariel/yolov5_quant_sample/b_data/labels/" # Replace with the folder containing YOLOv5 labels
    coco_json_output = "/home/user1/ariel/yolov5_quant_sample/b_data/annotations/Ben_annot.json"# Replace with the desired output JSON file path
    image_folder ="/home/user1/ariel/yolov5_quant_sample/b_data/images/"  # Replace with the folder containing the corresponding images

    yolov5_to_coco(yolov5_labels_path, coco_json_output, image_folder)

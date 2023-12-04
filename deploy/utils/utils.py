def calc_euclidean(box, ref_point):
    # calculate euclidean distance between box in ij format to the referance point
    dist = 0
    for i, _ in enumerate(box):
        dist += (box[i] - ref_point[i]) ** 2
    dist = dist ** (0.5)
    print('box is ' + str(box))
    print('ij are' + str(ref_point))
    print('distance is ' + str(dist))
    return dist

def xyxy_to_tlwh(box):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [top-left x1, y2, w, h] where xy1=top-left, xy2=bottom-right
    return box[0], box[1], box[2] - box[0], box[3] - box[1]

def xyxy_to_ij(boxes):
    # takes boxes in xywh format, convert to ij format, together with the eqiuvelant score
    boxes_ij = []
    for box in boxes:
        i = int((box[0] + box[2]) / 2)  # 1 , 3
        j = int((box[1] + box[3]) / 2)  # 2 , 4
        box_ij = [i, j, box[4]]  # 5
        print('box is ' + str(box))
        print('ij are' + str(box_ij))
        boxes_ij.append(box_ij)
    return boxes_ij


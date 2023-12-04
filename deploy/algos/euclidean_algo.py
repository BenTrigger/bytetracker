from ..utils.utils import xyxy_to_ij, xyxy_to_tlwh, calc_euclidean
import numpy as np

class Euclidean:
    def __init__(self):
        self.epsilon = 0.00001

    def get_best_pred_by_indication(self, boxes, ref_point, img_after_flip):
        # Inputs : boxes - list of boxes from ATR, ref_point - referance point in ij format
        # output: one box from ATR
        if len(boxes) > 0:
        #if boxes[0].numel():
            #boxes = self.check_label(boxes, img_after_flip)
            ij_boxes = xyxy_to_ij(boxes)
            box = self.score_calc(boxes, ij_boxes, ref_point)
        else:
            return None
        return xyxy_to_tlwh(box)  # As BoundingBox for rocx

    def score_calc(self, boxes, ij_boxes, ref_point):
        # calculate the score of each bbox considering ref point, then return the box with the maximal score
        final_scores = []
        for box in ij_boxes:
            print('box is ' + str(box))
            final_scores.append(box[2]/(calc_euclidean(box[0:2], ref_point)+self.epsilon))
            print('final score is ' + str((box[2]/(calc_euclidean(box[0:2], ref_point)+self.epsilon))))
        print('final scores are' + str(final_scores))
        return boxes[np.argmax(final_scores)]



    # def default_bbox(self, orig_imgsz, w, h):
    #     default_box = [] #orig_imgsz - original image's size [h, w, c]
    #     default_box.append(int(orig_imgsz[1]/2 - w/2))
    #     default_box.append(int(orig_imgsz[0]/2 - h/2))
    #     default_box.append(int(orig_imgsz[1]/2 + w/2))
    #     default_box.append(int(orig_imgsz[0]/2 + h/2))
    #     return default_box

    # def flip_handle(self, boxes, ref_point, image):
    #     if boxes[0].numel():
    #         boxes = self.check_label(boxes, image)
    #         ij_boxes = xyxy_to_ij(boxes)
    #         print('ij boxes are' + str(ij_boxes))
    #         box_chosen = self.score_calc(boxes ,ij_boxes, ref_point)
    #     else:
    #         print('No relevant boxes from ATR, choosing default ref point naive box!!')
    #         return None
    #     return box_chosen

    # def check_label(self, boxes, image):
    # # Check boxes are with label 0 - drones only!
    #     boxes_label = []
    #     for box in boxes:
    #         print('box is ' + str(box))
    #         box = box[0].cpu().detach().numpy()
    #         print('box after ' + str(box))
    #         if (box[5] == 0):
    #             boxes_label.append(box)
    #     return boxes_label


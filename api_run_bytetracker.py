from deploy.byte_tracker import ByteTracker
import cv2
from pathlib import Path
bt = ByteTracker()
imgs = []
for i in range(5):
    img_name = str(i)+'.jpg'
    #imgs.append(cv2.cvtColor(cv2.imread(Path('deploy') / img_name), cv2.COLOR_BGR2RGB))
    imgs.append(cv2.imread(Path('deploy') / img_name))
    bt.flip_treatment(imgs[i],[3840/2, 2160/2])
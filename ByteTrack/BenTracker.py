from yolox.tracker.byte_tracker import BYTETracker
tracker = BYTETracker(args)
for img im imgs:
    dets = detector(image)
    online_targets = tracker.update(dets, info_imgs, img_size)

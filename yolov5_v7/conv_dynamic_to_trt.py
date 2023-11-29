import torch
import torch_tensorrt

#model = torch.load('weights/Ben_tracker_model_3840_test.pt')
model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/Ben_tracker_model_3840_test.pt')
model.eval().cuda()
# Compile with static shapes
#inputs = torch_tensorrt.Input(shape=[1, 3, 224, 224], dtype=torch.float32)
# or compile with dynamic shapes
inputs = torch_tensorrt.Input(min_shape=[1, 3, 224, 224],
                              opt_shape=[4, 3, 224, 224],
                              max_shape=[8, 3, 224, 224],
                              dtype=torch.float32)
trt_gm = torch_tensorrt.compile(model, ir="dynamo", inputs)
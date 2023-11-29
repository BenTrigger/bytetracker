import copy
import time
import types
from speedster import optimize_model
import torch

model = torch.hub.load('/work/nebuly/optimization/speedster/yolov5', 'custom', path='yolov5/Ben_tracker_model_3840.pt', force_reload=True)
input_data = [((torch.randn(1, 3, 256, 256), ), torch.tensor([0])) for _ in range(100)]


# Run Speedster optimization
optimized_model = optimize_model(
    model,
    input_data=input_data,
    optimization_time="constrained",
    metric_drop_ths=0.05
)

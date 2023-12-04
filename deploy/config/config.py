import yaml
from pathlib import Path


def init_params(path="bytetracker_params.yaml", index=1):
    return get_yaml(path, index)


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def get_yaml(path="bytetracker_params.yaml", index=0):
    """
    get config yaml file - read and convert to struct
    :param path:path to yaml file
    :param index: 0 - HyperParams , 1 - OtherParams
    :return: struct of HyperParams (example - article_info follow bellow comments)
    """
    with open(path, "r") as yaml_file:
        data = yaml.load(yaml_file, Loader=yaml.FullLoader)
    # if index == 0:
    #     param = 'Yolo'
    # elif index == 1:
    #     param = 'ByteTracker'
    # elif index == 2:
    #     param = 'Stream'
    # else:
    #     return Exception("There Is No suck index: %s" % index)
    return Struct(**(data))#[index][param]))

#import argparse
# def get_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
#     parser.add_argument("--ByteTrackerFrames", type=int, default=1, help="test mot20.")
#     parser.add_argument("--track_thresh", type=float, default=0.6, help="tracking confidence threshold")
#     parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
#     parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
#     parser.add_argument("--min_track_thresh", type=float, default=0.1, help='threshold for minimum assignment')
#     opt = parser.parse_args()
#     return opt
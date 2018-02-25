import numpy as np
import cv2
import torch

class ToTensor(object):
    def __call__(self, sample):
        sample["img"] = torch.from_numpy(sample["img"].transpose((2, 0, 1))).float()
        sample["seq"] = torch.Tensor(sample["seq"]).int()
        return sample


class Resize(object):
    def __init__(self, size=(320, 32)):
        self.size = size

    def __call__(self, sample):
        sample["img"] = cv2.resize(sample["img"], self.size)
        return sample

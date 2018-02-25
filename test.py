import os
import cv2
from tqdm import tqdm
import click
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset.test_data import TestDataset
from dataset.data_transform import ToTensor, Resize
from models.model_loader import load_model
from torchvision.transforms import Compose

def pred_to_string(pred):
    seq = []
    for i in range(pred.shape[0]):
        label = np.argmax(pred[i])
        seq.append(label - 1)
    out = []
    for i in range(len(seq)):
        if len(out) == 0:
            if seq[i] != -1:
                out.append(seq[i])
        else:
            if seq[i] != -1 and seq[i] != seq[i - 1]:
                out.append(seq[i])
    out = ''.join(str(i) for i in out)
    return out

def test(net, data, cuda, visualize):
    data_loader = DataLoader(data, batch_size=1, num_workers=1, shuffle=False)

    count = 0
    tp = 0
    iterator = tqdm(data_loader)
    for sample in iterator:
        imgs = Variable(sample["img"])
        if cuda:
            imgs = imgs.cuda()
        pred = net(imgs).cpu().permute(1, 0, 2).data[0].numpy()
        out = pred_to_string(pred)
        gt = (sample["seq"][0].numpy() - 1).tolist()
        gt = ''.join(str(i) for i in gt)
        if out == gt:
            tp += 1
        count += 1
        if visualize:
            status = "pred: {}; gt: {}".format(out, gt)
            iterator.set_description(status)
            img = imgs[0].permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
            cv2.imshow("img", img)
            key = chr(cv2.waitKey() & 255)
            if key == 'q':
                break

    acc = tp / count
    return acc

@click.command()
@click.option('--num-classes', type=int, default=10, help='Number of classes')
@click.option('--seq-proj', type=str, default="10x20", help='Projection of sequence')
@click.option('--backend', type=str, default="resnet18", help='Backend network')
@click.option('--snapshot', type=str, default=None, help='Pre-trained weights')
@click.option('--input-size', type=str, default="320x32", help='Input size')
@click.option('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
@click.option('--visualize', type=bool, default=False, help='Visualize output')
def main(num_classes, seq_proj, backend, snapshot, input_size, gpu, visualize):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    cuda = True if gpu is not '' else False

    seq_proj = [int(x) for x in seq_proj.split('x')]
    net = load_model(num_classes, seq_proj, backend, snapshot, cuda).eval()
    input_size = [int(x) for x in input_size.split('x')]
    transform = Compose([
        Resize(size=(input_size[0], input_size[1])),
        ToTensor()
    ])
    data = TestDataset(transform=transform)
    acc = test(net, data, cuda, visualize)
    print("Accuracy: {}".format(acc))

if __name__ == '__main__':
    main()

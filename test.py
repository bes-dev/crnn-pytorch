import os
import cv2
import string
from tqdm import tqdm
import click
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset.test_data import TestDataset
from dataset.text_data import TextDataset
from dataset.collate_fn import text_collate
from dataset.data_transform import ToTensor, Resize
from models.model_loader import load_model
from torchvision.transforms import Compose

def test(net, data, abc, cuda, visualize):
    data_loader = DataLoader(data, batch_size=1, num_workers=1, shuffle=False, collate_fn=text_collate)

    count = 0
    tp = 0
    iterator = tqdm(data_loader)
    for sample in iterator:
        imgs = Variable(sample["img"])
        if cuda:
            imgs = imgs.cuda()
        out = net(imgs, decode=True)[0]
        gt = (sample["seq"].numpy() - 1).tolist()
        gt = ''.join(abc[i] for i in gt)
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
@click.option('--data-path', type=str, default=None, help='Path to dataset')
@click.option('--abc', type=str, default=string.digits+string.ascii_uppercase, help='Alphabet')
@click.option('--seq-proj', type=str, default="10x20", help='Projection of sequence')
@click.option('--backend', type=str, default="resnet18", help='Backend network')
@click.option('--snapshot', type=str, default=None, help='Pre-trained weights')
@click.option('--input-size', type=str, default="320x32", help='Input size')
@click.option('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
@click.option('--visualize', type=bool, default=False, help='Visualize output')
def main(data_path, abc, seq_proj, backend, snapshot, input_size, gpu, visualize):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    cuda = True if gpu is not '' else False

    seq_proj = [int(x) for x in seq_proj.split('x')]
    net = load_model(abc, seq_proj, backend, snapshot, cuda).eval()
    input_size = [int(x) for x in input_size.split('x')]
    transform = Compose([
        Resize(size=(input_size[0], input_size[1]))
    ])
    if data_path is not None:
        data = TextDataset(data_path=data_path, mode="test", transform=transform)
    else:
        data = TestDataset(transform=transform, abc=abc)
    acc = test(net, data, abc, cuda, visualize)
    print("Accuracy: {}".format(acc))

if __name__ == '__main__':
    main()

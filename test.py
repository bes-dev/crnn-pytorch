import os
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

def test(net, data, cuda):
    data_loader = DataLoader(data, batch_size=1, num_workers=1, shuffle=False)

    tp = 0
    for sample in tqdm(data_loader):
        imgs = Variable(sample["img"])
        if cuda:
            imgs = imgs.cuda()
        pred = net(imgs).cpu().permute(1, 0, 2).data[0].numpy()
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
        gt = (sample["seq"][0].numpy() - 1).tolist()
        out = ''.join(str(i) for i in out)
        gt = ''.join(str(i) for i in gt)
        if out == gt:
            tp += 1

    acc = tp / len(data)
    return acc

@click.command()
@click.option('--num-classes', type=int, default=10, help='Number of classes')
@click.option('--seq-proj', type=str, default="10x20", help='Projection of sequence')
@click.option('--backend', type=str, default="resnet18", help='Backend network')
@click.option('--snapshot', type=str, default=None, help='Pre-trained weights')
@click.option('--input-size', type=str, default="320x32", help='Input size')
@click.option('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
def main(num_classes, seq_proj, backend, snapshot, input_size, gpu):
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
    acc = test(net, data, cuda)
    print("Accuracy: {}".format(acc))

if __name__ == '__main__':
    main()

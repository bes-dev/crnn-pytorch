import os
import click
import string
import numpy as np
from tqdm import tqdm
from models.model_loader import load_model
from torchvision.transforms import Compose
from dataset.data_transform import Resize, Rotation, Translation, Scale
from dataset.test_data import TestDataset
from dataset.text_data import TextDataset
from dataset.collate_fn import text_collate
from lr_policy import StepLR

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch import Tensor
from torch.utils.data import DataLoader
from warpctc_pytorch import CTCLoss

from test import test

@click.command()
@click.option('--data-path', type=str, default=None, help='Path to dataset')
@click.option('--abc', type=str, default=string.digits+string.ascii_uppercase, help='Alphabet')
@click.option('--seq-proj', type=str, default="10x20", help='Projection of sequence')
@click.option('--backend', type=str, default="resnet18", help='Backend network')
@click.option('--snapshot', type=str, default=None, help='Pre-trained weights')
@click.option('--input-size', type=str, default="320x32", help='Input size')
@click.option('--base-lr', type=float, default=1e-3, help='Base learning rate')
@click.option('--step-size', type=int, default=500, help='Step size')
@click.option('--max-iter', type=int, default=6000, help='Max iterations')
@click.option('--batch-size', type=int, default=256, help='Batch size')
@click.option('--output-dir', type=str, default=None, help='Path for snapshot')
@click.option('--test-epoch', type=int, default=None, help='Test epoch')
@click.option('--test-init', type=bool, default=False, help='Test initialization')
@click.option('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
def main(data_path, abc, seq_proj, backend, snapshot, input_size, base_lr, step_size, max_iter, batch_size, output_dir, test_epoch, test_init, gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    cuda = True if gpu is not '' else False

    input_size = [int(x) for x in input_size.split('x')]
    transform = Compose([
        Rotation(),
        # Translation(),
        # Scale(),
        Resize(size=(input_size[0], input_size[1]))
    ])
    if data_path is not None:
        data = TextDataset(data_path=data_path, mode="train", transform=transform)
    else:
        data = TestDataset(transform=transform, abc=abc)
    seq_proj = [int(x) for x in seq_proj.split('x')]
    net = load_model(data.get_abc(), seq_proj, backend, snapshot, cuda)
    optimizer = optim.Adam(net.parameters(), lr = base_lr, weight_decay=0.0001)
    lr_scheduler = StepLR(optimizer, step_size=step_size, max_iter=max_iter)
    loss_function = CTCLoss()

    acc_best = 0
    epoch_count = 0
    while True:
        if (test_epoch is not None and epoch_count != 0 and epoch_count % test_epoch == 0) or (test_init and epoch_count == 0):
            print("Test phase")
            data.set_mode("test")
            net = net.eval()
            acc, avg_ed = test(net, data, data.get_abc(), cuda, visualize=False)
            net = net.train()
            data.set_mode("train")
            if acc > acc_best:
                if output_dir is not None:
                    torch.save(net.state_dict(), os.path.join(output_dir, "crnn_" + backend + "_" + str(data.get_abc()) + "_best"))
                acc_best = acc
            print("acc: {}\tacc_best: {}; avg_ed: {}".format(acc, acc_best, avg_ed))

        data_loader = DataLoader(data, batch_size=batch_size, num_workers=1, shuffle=True, collate_fn=text_collate)
        loss_mean = []
        iterator = tqdm(data_loader)
        iter_count = 0
        for sample in iterator:
            # for multi-gpu support
            if sample["img"].size(0) % len(gpu.split(',')) != 0:
                continue
            optimizer.zero_grad()
            imgs = Variable(sample["img"])
            labels = Variable(sample["seq"]).view(-1)
            label_lens = Variable(sample["seq_len"].int())
            if cuda:
                imgs = imgs.cuda()
            preds = net(imgs).cpu()
            pred_lens = Variable(Tensor([preds.size(0)] * batch_size).int())
            loss = loss_function(preds, labels, pred_lens, label_lens) / batch_size
            loss.backward()
            nn.utils.clip_grad_norm(net.parameters(), 10.0)
            loss_mean.append(loss.data[0])
            status = "epoch: {}; iter: {}; lr: {}; loss_mean: {}; loss: {}".format(epoch_count, lr_scheduler.last_iter, lr_scheduler.get_lr(), np.mean(loss_mean), loss.data[0])
            iterator.set_description(status)
            optimizer.step()
            lr_scheduler.step()
            iter_count += 1
        if output_dir is not None:
            torch.save(net.state_dict(), os.path.join(output_dir, "crnn_" + backend + "_" + str(data.get_abc()) + "_last"))
        epoch_count += 1

    return

if __name__ == '__main__':
    main()

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from myloss import dice_loss
from utils import *


def eval_net(net, dataset, gpu=False):
    tot = 0
    val_size = len(dataset)
    for (image, label) in dataset:
        x = np.array(image)
        y = np.array(label)

        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)

        x = x.permute(2, 0, 1).contiguous()  # transform to (C x H x W)

        x = x.view(1, 3, 256, 255)  # image (N x C x H x W)
        y = y.view(1, 1, 256, 255)  # mask (N x C x H x W)

        if gpu:
            x = Variable(x).cuda()
            y = Variable(y).cuda()
        else:
            x = Variable(x)
            y = Variable(y)

        x = normalize(x)  # normalize values to [0, 1]
        y = normalize(y)

        y_pred = net(x)  # feed into the net

        y_pred = y_pred.view(-1)  # make Tensor 1-dimensional
        probs = F.sigmoid(y_pred)

        y_flat = y.view(-1)
        dice = dice_loss(probs.float(), y_flat.float())  # calculate Dice-Coefficient

        tot += dice

    return tot / val_size
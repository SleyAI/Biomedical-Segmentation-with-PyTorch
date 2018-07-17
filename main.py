import sys
from optparse import OptionParser

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torchvision.transforms import Compose
import torchvision.transforms as transforms

from eval import eval_net
from unet import *
from utils import *


input_transform = Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    ])


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)


def train_net(net, epochs=1000, lr=0.0001, cp=True, gpu=True):

    #  Paths to data / validation directories
    data_dir = '/home/henning/PycharmProjects/Bachelor_Thesis/data/'
    val_dir = '/home/henning/PycharmProjects/Bachelor_Thesis/val/'
    dir_img = '/home/henning/PycharmProjects/Bachelor_Thesis/data/images.txt'
    dir_mask = '/home/henning/PycharmProjects/Bachelor_Thesis/data/masks.txt'
    dir_val_img = '/home/henning/PycharmProjects/Bachelor_Thesis/val/images.txt'
    dir_val_masks = '/home/henning/PycharmProjects/Bachelor_Thesis/val/masks.txt'
    dir_checkpoint = 'checkpoints/'

    #  Load train / validation data
    dataset = BreastCancerDataset(dir_img, dir_mask, data_dir, input_transform)
    val = BreastCancerDataset(dir_val_img, dir_val_masks, val_dir)

    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    criterion = nn.BCELoss()  # Binary Cross Entropy

    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        epoch_loss = 0
        i = 0

        #  Calculate Dice-Coefficient
        val_dice = eval_net(net, val, gpu)
        print('Validation Dice Coeff: {}'.format(val_dice))

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
            loss = criterion(probs.float(), y_flat.float())  # calculate loss
            epoch_loss += loss.item()
            i = i + 1

            """
            if i % 20 == 0:
                print('{0:d} --- loss: {1:.6f}'.format(i, loss.item()))
            """

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch finished ! Loss: {}'.format(epoch_loss / i))

        #  Save model after specific amount of epochs
        if cp and epoch % 50 == 0:
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP{}.pth'.format(epoch + 1))

            print('Checkpoint {} saved !'.format(epoch + 1))


"""Parser Section"""
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.000001,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')

    (options, args) = parser.parse_args()

    net = UNet(3, 1).apply(init_weights)  # Initialize Model

    if options.load:
        net.load_state_dict(torch.load(options.load))
        print('Model loaded from {}'.format(options.load))

    if options.gpu:
        net.cuda()
        cudnn.benchmark = True

    try:
        train_net(net, options.epochs, options.lr,
                  gpu=options.gpu)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

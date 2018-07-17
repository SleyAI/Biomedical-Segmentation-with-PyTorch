import argparse

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from unet import UNet
from utils import *
from utils.crf import dense_crf
from PIL import Image


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def predict_img(net, full_img, gpu=False):
    img = resize(full_img)
    img = np.array(img)
    img = torch.FloatTensor(img)

    x = img.permute(2, 0, 1).contiguous()  # transform to (C x H x W)

    x = x.view(1, 3, 256, 255)  # image (N x C x H x W)

    if gpu:
        with torch.no_grad():
            x = Variable(x).cuda()
    else:
        with torch.no_grad():
            x = Variable(x)

    x = normalize(x)  # normalize values to [0, 1]

    x = net(x)  # feed into the net

    x = F.sigmoid(x)
    x = F.upsample_bilinear(x, scale_factor=2).data[0][0].cpu().numpy()  # rescale the image to full size

    yy = dense_crf(np.array(full_img).astype(np.uint8), x)

    return yy > 0.5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'MODEL.pth')")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='filenames of ouput images')
    parser.add_argument('--cpu', '-c', action='store_true',
                        help="Do not use the cuda version of the net",
                        default=False)
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_false',
                        help="Do not save the output masks",
                        default=False)

    args = parser.parse_args()
    print("Using model file : {}".format(args.model))
    net = UNet(3, 1).apply(init_weights)
    if not args.cpu:
        print("Using CUDA version of the net, prepare your GPU !")
        net.cuda()
    else:
        net.cpu()
        print("Using CPU version of the net, this may be very slow")

    in_files = args.input
    out_files = []
    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        print("Error : Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    print("Loading model ...")
    net.load_state_dict(torch.load(args.model))
    print("Model loaded !")

    for i, fn in enumerate(in_files):
        print("\nPredicting image {} ...".format(fn))
        img = Image.open(fn)
        out = predict_img(net, img, not args.cpu)
        if not args.no_save:
            out_fn = out_files[i]
            result = Image.fromarray((out * 255).astype(np.uint8))
            result.save(out_files[i])
            print("Mask saved to {}".format(out_files[i]))

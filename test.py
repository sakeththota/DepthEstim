import torch
from utils import AverageMeter, DepthNorm, colorize
from loss import ssim
import torch.nn as nn
from data import getTrainingTestingData
from model import BModel, PTModel

def main():
    model = BModel().cuda()
    print("loading model")
    model.load_state_dict(torch.load("weights/baseline_gradient_ep_3_0.457.pth", map_location='cpu'))

    _, test_loader = getTrainingTestingData(batch_size=4)
    model.eval()
    sequential = test_loader
    sample_batched = next(iter(sequential))
    image = torch.autograd.Variable(sample_batched['image'].cuda())
    depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))
    depth_n = DepthNorm( depth )
    output = model(image)
    l1_criterion = nn.L1Loss()
    l_depth = l1_criterion(output, depth_n)
    print(f'l1 loss is {l_depth}')
    print(f'ssim is {ssim(output, depth_n, val_range = 1000.0 / 10.0)}') #ssim(output, depth_n, val_range = 1000.0 / 10.0))

if __name__ == '__main__':
    main()

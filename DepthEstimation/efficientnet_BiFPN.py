import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

from DepthEstimation.utils import NYUDataset, visualize_depth_map, visual_mask_fn, data_transform



class BiFPN(nn.Module):
  def __init__(self, fpn_sizes):
    super(BiFPN, self).__init__()

    P3_channels, P4_channels, P5_channels, P6_channels, P7_channels = fpn_sizes
    self.W_bifpn = 64

    # self.p6_td_conv  = nn.Conv2d(P6_channels, self.W_bifpn, kernel_size=3, stride=1, groups=self.W_bifpn, bias=True, padding=1)
    self.p6_td_conv = nn.Conv2d(P6_channels, self.W_bifpn, kernel_size=3, stride=1, bias=True, padding=1)
    self.p6_td_conv_2 = nn.Conv2d(self.W_bifpn, self.W_bifpn, kernel_size=3, stride=1, groups=self.W_bifpn, bias=True,
                                  padding=1)
    self.p6_td_act = nn.ReLU()
    self.p6_td_conv_bn = nn.BatchNorm2d(self.W_bifpn)
    self.p6_td_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
    self.p6_td_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)

    self.p5_td_conv = nn.Conv2d(P5_channels, self.W_bifpn, kernel_size=3, stride=1, bias=True, padding=1)
    self.p5_td_conv_2 = nn.Conv2d(self.W_bifpn, self.W_bifpn, kernel_size=3, stride=1, groups=self.W_bifpn, bias=True,
                                  padding=1)
    self.p5_td_act = nn.ReLU()
    self.p5_td_conv_bn = nn.BatchNorm2d(self.W_bifpn)
    self.p5_td_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
    self.p5_td_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)

    self.p4_td_conv = nn.Conv2d(P4_channels, self.W_bifpn, kernel_size=3, stride=1, bias=True, padding=1)
    self.p4_td_conv_2 = nn.Conv2d(self.W_bifpn, self.W_bifpn, kernel_size=3, stride=1, groups=self.W_bifpn, bias=True,
                                  padding=1)
    self.p4_td_act = nn.ReLU()
    self.p4_td_conv_bn = nn.BatchNorm2d(self.W_bifpn)
    self.p4_td_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
    self.p4_td_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)
    self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')

    self.p3_out_conv = nn.Conv2d(P3_channels, self.W_bifpn, kernel_size=3, stride=1, bias=True, padding=1)
    self.p3_out_conv_2 = nn.Conv2d(self.W_bifpn, self.W_bifpn, kernel_size=3, stride=1, groups=self.W_bifpn, bias=True,
                                   padding=1)
    self.p3_out_act = nn.ReLU()
    self.p3_out_conv_bn = nn.BatchNorm2d(self.W_bifpn)
    self.p3_out_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
    self.p3_out_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)

    # self.p4_out_conv = nn.Conv2d(P4_channels, self.W_bifpn, kernel_size=3, stride=1, bias=True, padding=1)
    self.p4_out_conv = nn.Conv2d(self.W_bifpn, self.W_bifpn, kernel_size=3, stride=1, groups=self.W_bifpn, bias=True,
                                 padding=1)
    self.p4_out_act = nn.ReLU()
    self.p4_out_conv_bn = nn.BatchNorm2d(self.W_bifpn)
    self.p4_out_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
    self.p4_out_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)
    self.p4_out_w3 = torch.tensor(1, dtype=torch.float, requires_grad=True)

    # self.p5_out_conv = nn.Conv2d(P5_channels,self.W_bifpn, kernel_size=3, stride=1, bias=True, padding=1)
    self.p5_out_conv = nn.Conv2d(self.W_bifpn, self.W_bifpn, kernel_size=3, stride=1, groups=self.W_bifpn, bias=True,
                                 padding=1)
    self.p5_out_act = nn.ReLU()
    self.p5_out_conv_bn = nn.BatchNorm2d(self.W_bifpn)
    self.p5_out_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
    self.p5_out_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)
    self.p5_out_w3 = torch.tensor(1, dtype=torch.float, requires_grad=True)
    self.p4_downsample = nn.MaxPool2d(kernel_size=2)

    # self.p6_out_conv = nn.Conv2d(P6_channels, self.W_bifpn, kernel_size=3, stride=1, bias=True, padding=1)
    self.p6_out_conv = nn.Conv2d(self.W_bifpn, self.W_bifpn, kernel_size=3, stride=1, groups=self.W_bifpn, bias=True,
                                 padding=1)
    self.p6_out_act = nn.ReLU()
    self.p6_out_conv_bn = nn.BatchNorm2d(self.W_bifpn)
    self.p6_out_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
    self.p6_out_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)
    self.p6_out_w3 = torch.tensor(1, dtype=torch.float, requires_grad=True)
    # self.p4_downsample= nn.MaxPool2d(kernel_size=2)

    self.p7_out_conv = nn.Conv2d(P7_channels, self.W_bifpn, kernel_size=3, stride=1, bias=True, padding=1)
    self.p7_out_conv_2 = nn.Conv2d(self.W_bifpn, self.W_bifpn, kernel_size=3, stride=1, groups=self.W_bifpn, bias=True,
                                   padding=1)
    self.p7_out_act = nn.ReLU()
    self.p7_out_conv_bn = nn.BatchNorm2d(self.W_bifpn)
    self.p7_out_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
    self.p7_out_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)

  def forward(self, inputs):
    epsilon = 0.0001
    P3, P4, P5, P6, P7 = inputs
    # print ("Input::", P3.shape, P4.shape, P5.shape, P6.shape, P7.shape)
    # P6_td = self.p6_td_conv((self.p6_td_w1 * P6 ) /
    #                         (self.p6_td_w1 + epsilon))

    P7_td = self.p7_out_conv(P7)

    P6_td_inp = self.p6_td_conv(P6)
    P6_td = self.p6_td_conv_2((self.p6_td_w1 * P6_td_inp + self.p6_td_w2 * P7_td) /
                              (self.p6_td_w1 + self.p6_td_w2 + epsilon))
    # P6_td = self.p6_td_conv_2(P6_td_inp)
    P6_td = self.p6_td_act(P6_td)
    P6_td = self.p6_td_conv_bn(P6_td)

    P5_td_inp = self.p5_td_conv(P5)
    # print (P5_td_inp.shape, P6_td.shape)
    P5_td = self.p5_td_conv_2((self.p5_td_w1 * P5_td_inp + self.p5_td_w2 * P6_td) /
                              (self.p5_td_w1 + self.p5_td_w2 + epsilon))
    P5_td = self.p5_td_act(P5_td)
    P5_td = self.p5_td_conv_bn(P5_td)

    # print (P4.shape, P5_td.shape)
    P4_td_inp = self.p4_td_conv(P4)
    P4_td = self.p4_td_conv_2((self.p4_td_w1 * P4_td_inp + self.p4_td_w2 * self.p5_upsample(P5_td)) /
                              (self.p4_td_w1 + self.p4_td_w2 + epsilon))
    P4_td = self.p4_td_act(P4_td)
    P4_td = self.p4_td_conv_bn(P4_td)

    P3_td = self.p3_out_conv(P3)
    P3_out = self.p3_out_conv_2((self.p3_out_w1 * P3_td + self.p3_out_w2 * P4_td) /
                                (self.p3_out_w1 + self.p3_out_w2 + epsilon))
    P3_out = self.p3_out_act(P3_out)
    P3_out = self.p3_out_conv_bn(P3_out)

    # print (P4_td.shape, P3_out.shape)

    P4_out = self.p4_out_conv(
      (self.p4_out_w1 * P4_td_inp + self.p4_out_w2 * P4_td + self.p4_out_w3 * P3_out)
      / (self.p4_out_w1 + self.p4_out_w2 + self.p4_out_w3 + epsilon))
    P4_out = self.p4_out_act(P4_out)
    P4_out = self.p4_out_conv_bn(P4_out)

    P5_out = self.p5_out_conv(
      (self.p5_out_w1 * P5_td_inp + self.p5_out_w2 * P5_td + self.p5_out_w3 * self.p4_downsample(P4_out))
      / (self.p5_out_w2 + self.p5_out_w3 + epsilon))
    P5_out = self.p5_out_act(P5_out)
    P5_out = self.p5_out_conv_bn(P5_out)

    P6_out = self.p6_out_conv((self.p6_out_w1 * P6_td_inp + self.p6_out_w2 * P6_td + self.p6_out_w3 * (P5_out))
                              / (self.p6_out_w1 + self.p6_out_w2 + self.p6_out_w3 + epsilon))
    P6_out = self.p6_out_act(P6_out)
    P6_out = self.p6_out_conv_bn(P6_out)

    P7_out = self.p7_out_conv_2((self.p7_out_w1 * P7_td + self.p7_out_w2 * P6_out) /
                                (self.p7_out_w1 + self.p7_out_w2 + epsilon))
    P7_out = self.p7_out_act(P7_out)
    P7_out = self.p7_out_conv_bn(P7_out)

    return [P3_out, P4_out, P5_out, P6_out, P7_out]


class EfficientNet(nn.Module):
  def __init__(self):
      super(EfficientNet, self).__init__()
      encoder = efficientnet_b0(pretrained=True)
      features = encoder.features
      num_of_sequences = len(features)
      self.layer1 = features[:num_of_sequences-4]
      self.layer2 = features[num_of_sequences-4]
      self.layer3 = features[num_of_sequences-3]
      self.layer4 = features[num_of_sequences-2]
      self.layer5 = features[num_of_sequences-1]

  def get_features(self, x):
    x1 = self.layer1(x)
    x2 = self.layer2(x1)
    x3 = self.layer3(x2)
    x4 = self.layer4(x3)
    x5 = self.layer5(x4)
    return x1,x2,x3,x4,x5

class EfficientNetBiFPN(nn.Module):
  def __init__(self):
    super(EfficientNetBiFPN, self).__init__()
    self.resnet_encoder = EfficientNet()
    self.bifpn = BiFPN([80, 112, 192, 320, 1280])

  def forward(self, data):
    x1, x2, x3, x4, x5 = self.resnet_encoder.get_features(data[0])
    return self.bifpn([x1, x2, x3, x4, x5])


data = NYUDataset(
  annotations_file='../../data/nyu_samples/nyu2_test.csv',
  img_dir='../../data/nyu_samples/nyu2_test',
  transform=data_transform,
  target_transform=visual_mask_fn
)
dataloader = torch.utils.data.DataLoader(data, batch_size=1, num_workers=2, shuffle=True)

visualize_depth_map(data, 0)
encoder = EfficientNetBiFPN()
out = encoder(next(iter(dataloader)))
print(out)
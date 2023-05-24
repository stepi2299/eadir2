import torch
import torch.nn as nn
from torch.functional import F


class LinearBlock_c(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        feature_size=3,
        num_inner_layers=1,
        kernel_size=3,
        padding=0,
        stride=1,
        mode="infer",
    ):
        super(LinearBlock_c, self).__init__()
        # Params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_size = feature_size
        self.mode = mode

        # expansion with kx,ky kernel and then project to out_filters using 1x1
        kernel_size = [kernel_size, kernel_size]
        self.kx, self.ky = kernel_size

        # Learnable Collapse Conv's

        conv1 = nn.Conv2d(in_channels, feature_size, kernel_size, padding="valid")

        conv2 = nn.Conv2d(feature_size, out_channels, 1, padding="valid")

        self.collapsed_weights = None

        # Define Collapse Block
        self.collapse = nn.Sequential(conv1, conv2)
        self.build()

    def build(self):
        # shape: (in_filters,in_filters)
        delta = torch.eye(self.in_channels)

        # expanded shape:(in_filters, 1, 1, in_filters)
        delta = torch.unsqueeze(torch.unsqueeze(delta, 1), 1)

        # padded shape: (in_filters, kx, ky, in_filters)
        self.delta = nn.functional.pad(
            delta,
            pad=(0, 0, self.kx - 1, self.kx - 1, self.kx - 1, self.kx - 1, 0, 0),
            mode="constant",
        )
        print("delta", self.delta.shape)

        # Calculate Residual
        kernel_dim = [self.kx, self.ky, self.in_channels, self.out_channels]
        residual = torch.zeros(kernel_dim)

        if self.in_channels == self.out_channels:
            mid_kx = int(self.kx / 2)
            mid_ky = int(self.ky / 2)

            for out_ch in range(self.out_channels):
                residual[mid_kx, mid_ky, out_ch, out_ch] = 1.0

        # Ensure the Value isn't trainable
        self.residual = residual

    def forward(self, inputs):
        if self.mode == "train" or (self.collapsed_weights is None):
            # Run Through Conv2D's - online linear collapse
            print("delta2", self.delta.shape)
            wt_tensor = self.collapse(self.delta)

            # reverse order of elements in 1,2 axes
            wt_tensor = torch.flip(wt_tensor, [1, 2])

            # (in_filters, kx, ky, out_filters) -> (kx, ky, in_filters, out_filters)
            wt_tensor = wt_tensor.permute(1, 2, 0, 3)

            # Direct-residual addition
            # when in_filters != self.out_filters, this is just zeros
            wt_tensor += self.residual

            if self.mode == "infer":
                # store collapsed weights in the first inferece, won't need to collapse again
                self.collapsed_weights = wt_tensor
                # remove references to uncollapsed variables
                self.collapse = None

        else:
            # use pre-collapsed weights
            wt_tensor = self.collapsed_weights

        # Output - the actual conv2d
        out = F.conv2d(inputs, wt_tensor, stride=1, padding="same")

        return out


a = LinearBlock_c(feature_size=11, kernel_size=3, in_channels=5, out_channels=3)
data = torch.randn(3, 3, 224, 224)
a.train()
a(data)
a.eval()
a(data)
a.collapsed_weights
print(a.collapsed_weights)

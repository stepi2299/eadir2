from exercises.semantic_segmentation.data_preparation import Dataset
from exercises.semantic_segmentation.FFN import Up_Head
import torch

dataset = Dataset("/home/stepi2299/repo/eadir2/data/seg_data")
img, seg = dataset[0]
img.show()
seg.show()

up_A = Up_Head(input_channels=[256, 512, 1024, 2048], middle_channels=[64, 128, 256, 512], out_channels=[128, 128, 128, 128])
up_B = Up_Head(input_channels=[256, 512, 1024, 2048], middle_channels=[64, 128, 128, 256], out_channels=[96, 96, 64, 32])
up_C = Up_Head(input_channels=[256, 512, 1024, 2048], middle_channels=[128, 128, 128, 128], out_channels=[128, 16, 16, 16])

c1 = torch.randn([1, 256, 64, 64])
c2 = torch.randn([1, 512, 64, 64])
c3 = torch.randn([1, 1024, 32, 32])
c4 = torch.randn([1, 2048, 16, 16])

out_A = up_A([c1, c2, c3, c4])
print("output A: ", out_A)

out_B = up_B([c1, c2, c3, c4])
print("output B: ", out_B)

out_C = up_C([c1, c2, c3, c4])
print("output C: ", out_C)

up_A_with_inv_residuals = Up_Head(input_channels=[256, 512, 1024, 2048], middle_channels=[64, 128, 256, 512], out_channels=[128, 128, 128, 128], inv_res=True)
out_A_res = up_A_with_inv_residuals([c1, c2, c3, c4])
print("output inverted residual with A: ", out_A_res)

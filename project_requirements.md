#TASKS
1.Create the Dataset object that properly load the input image and target mask. Save the class definition in the git repo.
display a few images without data augmentations, use some proper color map from matplotlib for better visualization.
Remember for training, you need to preprocess the target mask with the maximum depth.

2.Use a pretrained efficient-b0 as backbone(encoder) and bifpn as neck(decoder).
You need only to use features from resolution levels of 56x56, 28x28, 14x14 given inputs as 224x224. Which means that
from decoder, you also need the features of resolution 56x56, 28x28, 14x14. Then you need to upsample the 28x28 and 14x14 features to 56x56, concatenate them with 56x56 features, pass through another convolution layers to form 1 channel 56x56 data as final output, 
which means you should resize the target mask to 56x56 using bilinear interpolation.

3. Train with MSE loss function. You need 3 runs with different LR and 2 runs with data augmentation and without.

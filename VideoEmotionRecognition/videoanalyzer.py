from facenet_pytorch import MTCNN
from torch import nn
import torch
import torchvision.models as models
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.io
import os


VIDEONAME = "moja_twarz.mp4"


class FramesPredictionDataset(Dataset):
    def __init__(self, frames, transform=None):
      self.frames=frames
      self.detector=MTCNN()
      self.transform=transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
      frame = self.frames[idx]
      box1 = self.detector.detect(frame)
      box = box1[0]
      x, y, x2, y2 = box[0].astype("int16")
      crop_image = frame.numpy()[y:y2, x:x2]
      if self.transform:
        crop_image = self.transform(crop_image)
      return crop_image


class VideoNet:
    def __init__(self):
        self.mtcnn = MTCNN(image_size=224)
        self.feature_extractor = models.resnet18()
        self.feature_extractor.fc = nn.Linear(512, 7)
        self.feature_extractor.load_state_dict(torch.load('../videos/emotion.pth', map_location=torch.device('cpu')))
        self.lstm = nn.LSTM(input_size=7, hidden_size=128)
        self.h0 = torch.randn(1, 128)
        self.c0 = torch.randn(1, 128)
        self.fc = nn.Linear(128, 8)

    def _crop_face(self, frame):
        try:
            box = self.mtcnn.detect(frame)[0]
            x, y, x2, y2 = box[0].astype("int16")
            return frame.numpy()[y:y2, x:x2]
        except Exception as e:
            print(f"Exception: {e}")

    def forward(self, frame):
        features = self.feature_extractor(frame)
        vec, (hn, cn) = self.lstm(features, (self.h0, self.c0))
        return self.fc(vec)


frames = torchvision.io.read_video(os.path.join("../videos", VIDEONAME))[0]

transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
test_data = FramesPredictionDataset(frames, transform_val)
train_loader = DataLoader(test_data, batch_size=1, num_workers=1)
videonet = VideoNet()
for frame in train_loader:
    out = videonet.forward(frame)
    print(torch.argmax(out))
print(f"Final label: {torch.argmax(out)}, final vector: {out}",)




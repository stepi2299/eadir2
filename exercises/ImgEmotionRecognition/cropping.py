import torchvision.io
from facenet_pytorch import MTCNN
import cv2 as cv
import os


def crop_images(video_name, new_folder):
    frames = torchvision.io.read_video(os.path.join("../videos", video_name))[0]
    mtcnn = MTCNN(image_size=224)
    os.makedirs(new_folder, exist_ok=True)
    for i, frame in enumerate(frames):
        try:
            box = mtcnn.detect(frame)[0]
            x, y, x2, y2 = box[0].astype("int16")
            face = frame.numpy()[y:y2, x:x2]
            cv.imwrite(os.path.join(new_folder, f"{i}.jpg"), face)
        except Exception as e:
            print(f"Exception for {i}: {e}")
    print(f"Videao has {i} frames")

crop_images("moja_twarz.mp4", "face")





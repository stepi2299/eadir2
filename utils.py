import gdown
import os


def download(url, output):
    gdown.download(url, os.path.join("videos", output))
import glob
import random
from notebooks.utils import compare_images, enhance_image
from PIL import Image

dataset_path = "../yolo-testing/datasets/exDark-yolo"
filename = random.sample(glob.glob(f"{dataset_path}/images/train/*.jpg"), 1)[0]
image_out = enhance_image(filename)
compare_images(Image.open(filename), image_out)

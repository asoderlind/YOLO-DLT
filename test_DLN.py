import glob
import random
from notebooks.utils import compare_images, enhance_image, get_dln_model
from PIL import Image

dataset_path = "../yolo-testing/datasets/exDark-yolo"
filename = random.sample(glob.glob(f"{dataset_path}/images/train/*.jpg"), 1)[0]
filename = "../yolo-testing/datasets/exDark-yolo/images/train/2015_03267.jpg"
# filename = "../yolo-testing/datasets/exDark-yolo/images/train/2015_01388.jpg"
print(filename)
dln = get_dln_model()
image_out = enhance_image(filename, dln)
compare_images(Image.open(filename), image_out)

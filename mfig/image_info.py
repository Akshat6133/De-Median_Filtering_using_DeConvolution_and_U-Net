import numpy as np
from scipy.ndimage import median_filter
from PIL import Image
import matplotlib.pyplot as plt
from os import listdir


raw_img_path: str = "./../pokemon_dataset/images/"
mf_img_path: str = "./../mf_dataset/"
image_list: list = listdir(raw_img_path)

image_name = image_list[0]
image_full_path = raw_img_path + image_name
image = Image.open(image_full_path)

from PIL import Image
import numpy as np

# Load an image
image = Image.open("/home/akshat/projects/dip/mfig/filtered_image_gray.png")

# Print the image mode (shows format like RGB, L for grayscale, etc.)
print(f"Image mode: {image.mode}")

# Convert to a numpy array
img_array = np.array(image)

# Print the dtype of the image (this gives us the bit depth)
print(f"Image data type: {img_array.dtype}")

# If it's a color image, check the shape to ensure it's RGB or RGBA
if len(img_array.shape) == 3:
    print(f"Image shape: {img_array.shape} (height, width, channels)")
else:
    print(f"Image shape: {img_array.shape} (height, width)")

# Check the bit depth based on the dtype
if img_array.dtype == np.uint8:
    print("The image has 8-bit depth per channel (for each color channel).")
elif img_array.dtype == np.uint16:
    print("The image has 16-bit depth per channel (for each color channel).")
elif img_array.dtype == np.float32 or img_array.dtype == np.float64:
    print("The image might have floating point data, often used in higher bit depths.")
else:
    print(f"Unrecognized data type: {img_array.dtype}")


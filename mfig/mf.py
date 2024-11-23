from os import listdir
import numpy as np
from scipy.ndimage import median_filter
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from skimage.color import rgba2rgb, rgb2gray

# Set paths
raw_img_path: str = "./../pokemon_dataset/images/"
mf_img_path: str = "./../mf_dataset/"
gray_img_path: str = "./../grayscale_dataset/"

img_list: list = listdir(raw_img_path)

# Load image
# img_name = img_list[0]
# img_full_path = raw_img_path + img_name

def rgba2grayscale(raw_img_path, img_name, op_path):
    img_full_path = raw_img_path + img_name
    img_rgba = Image.open(img_full_path)
    rgb_img = rgba2rgb(np.array(img_rgba))  # Convert RGBA to RGB (3 channels)
    gray_img = rgb2gray(rgb_img)  # Convert RGB to grayscale
    plt.axis('off')
    plt.imshow(gray_img, cmap='gray')  # Use 'gray' colormap for grayscale
    output_path = op_path + img_name
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    # print(f"Saved grayscale image to {output_path}")
    return gray_img  # Return the grayscale image for further processing

def mf(src_img_path, img_name, op_path):
    # First, load the grayscale image
    gray_img = rgba2grayscale(src_img_path, img_name, gray_img_path)
    
    # Apply median filter to the grayscale image
    mf_sample = median_filter(gray_img, size=3)
    
    # Plot and save the filtered grayscale image
    plt.imshow(mf_sample, cmap='gray')  # Use 'gray' colormap for grayscale
    plt.axis('off')
    output_path = op_path + img_name
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)

    # print(f"Saved the filtered grayscale image to {output_path}")

# rgba2grayscale(raw_img_path, img_name , gray_img_path)
# mf(raw_img_path, img_name, mf_img_path)

n: int = len(img_list)
for i in range(n):
    img_name = img_list[i]
    mf(raw_img_path, img_name, mf_img_path)
    print(f"{i}/{n}")



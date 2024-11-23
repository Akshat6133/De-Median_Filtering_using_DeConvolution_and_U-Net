import cv2
import numpy as np
from os import listdir

mse_unet_ls:list = []
mse_mfdc_ls:list = []

grayscale_img_path: str = "./grayscale_dataset/"
mfdc_img_path: str = "./mfdc_dataset/"

img_names = listdir(grayscale_img_path)
n:int = len(img_names)

def calculate_mse_gpu(path1, path2):
    image1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

    image1 = image1.astype('float32')
    image2 = image2.astype('float32')

    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions")
    mse = np.mean((image1 - image2) ** 2)
    
    return mse

for i in range(n):
    img_name = img_names[i]
    gray_img = grayscale_img_path + img_name 
    mfdc_img = mfdc_img_path + img_name 
    mse = calculate_mse_gpu(gray_img,  mfdc_img)
    mse_mfdc_ls.append(mse)
    print("mse: ", mse)

    print(img_name, i , "/", n)

mean_mse_mfdc = np.mean(mse_mfdc_ls)

print(f"mse_mfdc: {mean_mse_mfdc}")
print(f"min mse_mfdc: {min(mse_mfdc_ls)}")
print(f"max mse_mfdc: {max(mse_mfdc_ls)}")



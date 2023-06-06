import numpy as np
import torch
import torch.nn as nn
import cv2

def lr_generation(ratio, path, save_path, blur_size = 3, blur_sigma = 0.5, noise_sigma = 10):
    print("Opening", path)
    X_ori = cv2.imread(path)
    X_ori = X_ori.astype(np.float32)
    height, width, _ = X_ori.shape

    H_kernel = cv2.getGaussianKernel(blur_size, blur_sigma) * cv2.getGaussianKernel(blur_size, blur_sigma).T

    temp1 = cv2.filter2D(X_ori, -1, H_kernel)

    size_decrease = (width//ratio, height//ratio)
    temp2 = cv2.resize(temp1, size_decrease).astype(np.float32)

    gauss = np.random.normal(0, noise_sigma, (height//ratio, width//ratio, 3)).astype(np.float32)
    temp3 = gauss + temp2

    cv2.imwrite(save_path, temp3)
    print("Saved to"+save_path)

import os
def main():
    ratio = 8
    path = "1.png"
    for image_name in os.listdir("./inputs_old/ground_truth/"):
        lr_generation(ratio, "./inputs_old/ground_truth/"+image_name, "./inputs/"+image_name+"_lr.jpg", blur_size = 3, blur_sigma = 0.5, noise_sigma = 10)


if __name__ == '__main__':
    main()








import os

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def evaluate(real: cv2.Mat, sr_result: cv2.Mat) -> dict[str, float]:
    '''
    An evaluation function that calculates Peak SNR and sturctural similarity.
    '''
    if len(real.shape)==2: # Real image is processed as grayscale
        sr_result = cv2.cvtColor(sr_result, cv2.COLOR_BGR2GRAY)
    if real.shape!=sr_result.shape:
        print("Shape not equal: %s, %s. Resize to equal shape %s. "%(real.shape, sr_result.shape, sr_result.shape))
        real = cv2.resize(real, dsize=(sr_result.shape[1], sr_result.shape[0]))
    psnr = peak_signal_noise_ratio(real, sr_result)
    print("Peak SNR: %f"%(psnr))
    ssim = structural_similarity(real, sr_result, channel_axis=2)
    print("Structural similarity: %f"%(ssim))
    l1 = np.mean(np.abs(sr_result-real))
    print("L1 loss: %f"%(l1))
    l2 = np.mean((sr_result-real)**2)
    print("L2 loss: %f"%(l2))
    return  psnr, ssim, l1, l2

if __name__ == "__main__":
    images_gt = []
    images_original_output = []
    images_new_output = []
    results = {
        "psnr1": [],
        "psnr2": [],
        "psnr3": [],
        "ssim1": [],
        "ssim2": [],
        "ssim3": [],
        "l1_1": [],
        "l1_2": [],
        "l1_3": [],
        "l2_1": [],
        "l2_2": [],
        "l2_3": [],
    }
    for image_name in os.listdir("./inputs_old/ground_truth/"):
        image_real_name, image_ext = os.path.splitext(image_name)
        print("="*30)
        print("Image name: %s"%image_name)
        image_gt = cv2.imread("./inputs_old/ground_truth/"+image_name)
        image_output_original = cv2.imread("./results_xhz/RealESRGAN_x4plus.pth/"+image_real_name+"_lr_out"+image_ext)
        psnr1, ssim1, l1_1, l2_1 = evaluate(image_gt, image_output_original)
        image_output_new = cv2.imread("./results_xhz/net_g_10000.pth/"+image_real_name+"_lr_out"+image_ext)
        psnr2, ssim2, l1_2, l2_2 = evaluate(image_gt, image_output_new)
        image_lr = cv2.imread("./inputs_old/low_resolution/"+image_real_name+"_lr"+image_ext)
        psnr3, ssim3, l1_3, l2_3 = evaluate(image_gt, cv2.resize(image_lr, dsize=(image_gt.shape[1], image_gt.shape[0])))
        results["psnr1"].append(psnr1)
        results["psnr2"].append(psnr2)
        results["psnr3"].append(psnr3)
        results["ssim1"].append(ssim1)
        results["ssim2"].append(ssim2)
        results["ssim3"].append(ssim3)
        results["l1_1"].append(l1_1)
        results["l1_2"].append(l1_2)
        results["l1_3"].append(l1_3)
        results["l2_1"].append(l2_1)
        results["l2_2"].append(l2_2)
        results["l2_3"].append(l2_3)
    print("\n"+"="*20+" Results: "+"="*20)
    print("Peak SNR: %f VS %f VS %f"%(
        np.mean(results["psnr1"]), np.mean(results["psnr2"]), np.mean(results["psnr3"])))
    print("Structural similarity: %f VS %f VS %f"%(
        np.mean(results["ssim1"]), np.mean(results["ssim2"]), np.mean(results["ssim3"])))
    print("L1 loss: %f VS %f VS %f"%(
        np.mean(results["l1_1"]), np.mean(results["l1_2"]), np.mean(results["l1_3"])))
    print("L2 loss: %f VS %f VS %f"%(
        np.mean(results["l2_1"]), np.mean(results["l2_2"]), np.mean(results["l2_3"])))
import csv
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from skimage.metrics import structural_similarity as ssim
import argparse
import imutils
import cv2

import torch
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer
from PIL import Image


# from scipy import linalg
# from keras.applications.inception_v3 import InceptionV3
# from keras.applications.inception_v3 import preprocess_input


def calculate_fid(image1_path, image2_path):
# Load InceptionV3 model without top layers
    # model = InceptionV3(include_top=False, pooling='avg')

    # # Load and preprocess images
    # image1 = load_image(image1_path)
    # image2 = load_image(image2_path)

    # # Get predictions from InceptionV3 model
    # features1 = model.predict(image1)
    # features2 = model.predict(image2)

    # # Calculate mean and covariance
    # mu1, sigma1 = np.mean(features1, axis=0), np.cov(features1, rowvar=False)
    # mu2, sigma2 = np.mean(features2, axis=0), np.cov(features2, rowvar=False)

    # # Calculate squared Frobenius norm between means
    # fid_score = np.sum((mu1 - mu2)**2)

    # # Calculate square root of product of covariances
    # cov_sqrt = sqrtm(sigma1.dot(sigma2))

    # # Check for imaginary numbers
    # if np.iscomplexobj(cov_sqrt):
    #     cov_sqrt = cov_sqrt.real

    # # Calculate FID score
    # fid_score += np.trace(sigma1 + sigma2 - 2*cov_sqrt)
    fid_score = 0
    return fid_score 

def calculate_clip_score (image1_path, image2_path):
    # Function to calculate CLIP distance
    # Load the CLIP model
    model_ID = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_ID)
    preprocess = CLIPImageProcessor.from_pretrained(model_ID)

    # image_a =Image.open(image1_path)["pixel_values"]
    image_a =Image.open(image1_path)
    image_a = preprocess(image_a, return_tensors="pt")

    # image_b =Image.open(image2_path)["pixel_values"]
    image_b =Image.open(image2_path)
    image_b = preprocess(image_b, return_tensors="pt")
    with torch.no_grad():
        embedding_a = model.get_image_features(image_a)
        embedding_b = model.get_image_features(image_b)

    # Calculate the cosine similarity between the embeddings
    similarity_score = torch.nn.functional.cosine_similarity(embedding_a, embedding_b)

    clip_score = similarity_score # Calculate CLIP distance
    return clip_score

def calculate_psnr(image1_path, image2_path):
    original = cv2.imread(image1_path) 
    compressed = cv2.imread(image2_path, 1) 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr_score = 20 * log10(max_pixel / sqrt(mse)) 

    return psnr_score

def calculate_ssim(image1_path, image2_path):
    # Function to calculate SSIM score
    imageA = cv2.imread(image1_path)
    imageB = cv2.imread(image2_path)
    print(f"Shape of im1: {imageA.shape}")
    print(f"Shape of im2: {imageB.shape}")

    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    (ssim_score, diff) = ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    
    return ssim_score

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <image1_path> <image2_path>")
        sys.exit(1)

    image1_path = sys.argv[1]
    image2_path = sys.argv[2]
    
    # fid_score = calculate_fid(image1_path, image2_path)
    # print(fid_score)    
    # clip_score = calculate_clip_score(image1_path, image2_path)
    # print(clip_score)
    # psnr_score = calculate_psnr(image1_path, image2_path)
    # print(psnr_score)
    ssim_score = calculate_ssim(image1_path, image2_path)
    print(ssim_score)
    # Output scores to CSV file
    with open("experiment_results.csv", "w", newline="") as csvfile:
        fieldnames = ["Metric", "Score"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow({"Metric": "FID", "Score": fid_score})
        writer.writerow({"Metric": "CLIP", "Score": clip_score})
        writer.writerow({"Metric": "PSNR", "Score": psnr_score})
        writer.writerow({"Metric": "SSIM", "Score": ssim_score})

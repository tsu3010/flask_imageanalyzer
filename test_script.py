import sys
import os
import glob
import re
import numpy as np
import pandas as pd
import cv2
from PIL import Image as IMG
from skimage import feature
from skimage.restoration import estimate_sigma

img_path = r'D:\DataScience\Image Classification\all\train\Blur-cat.10087.jpg'


def estimate_blurriness(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(image, cv2.CV_64F).var()
    return fm


def compute_quality(blur, threshold1=50, threshold2=100):
    blur_list = [blur.tolist()]
    blur_df = pd.DataFrame({'blur': blur_list})
    blur_df['labels'] = pd.cut(blur_df.blur, [0, threshold1, threshold2, float("inf")], labels=['Poor', 'Average', 'Good'])
    return blur_df['labels'].iloc[0]


def font_color(quality):
    if quality == 'Poor':
        col = 'red'
    elif quality == 'Average':
        col = 'orange'
    else:
        col = 'green'
    return col


blur = estimate_blurriness(img_path)
qual = compute_quality(blur)
print("Blur Value : %s" %(blur))
print("Image Quality : %s " % (qual))
print(font_color(qual))

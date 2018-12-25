import sys
import os
import glob
import re
import numpy as np
import cv2
from PIL import Image as IMG
from skimage import feature
from skimage.restoration import estimate_sigma
from flask import Flask, redirect, url_for, request, render_template, send_from_directory


# Define a flask app
app = Flask(__name__)

## Iniitialize   folder names ###
dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

# Use Image path to make computation


def estimate_blurriness(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(image, cv2.CV_64F).var()
    return fm


def estimate_uniformity(img_path):
    im = IMG.open(img_path)
    im_array = np.asarray(im.convert(mode='L'))
    edges_sigma1 = feature.canny(im_array, sigma=3)
    apw = (float(np.sum(edges_sigma1)) / (im.size[0] * im.size[1]))
    return apw


def estimate_noise(img_path):
    image = cv2.imread(img_path)
    return estimate_sigma(image, multichannel=True, average_sigmas=True)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():

    if request.method == 'GET':
        return render_template('index.html')
    else:
        # Get the file from post request
        file = request.files['image']
        # Save the file to ./uploads
        full_name = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(full_name)
        # Make feature computation
        blur = str(round(estimate_blurriness(full_name), 2))
        apw = str(round(estimate_uniformity(full_name), 2))
        noise = str(round(estimate_noise(full_name), 2))
    return render_template('predict.html', image_file_name=file.filename, blur_value=blur, apw_value=apw, noise_value=noise)


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/help', methods=['GET'])
def help():
    return render_template('help.html')


if __name__ == '__main__':
    app.run(debug=True)

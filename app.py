import sys
import os
import glob
import re
import numpy as np
import cv2
from flask import Flask, redirect, url_for, request, render_template, send_from_directory


# Define a flask app
app = Flask(__name__)

## Iniitialize   folder names ###
dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

# Use Image path to make computation


def image_feature_predict(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(image, cv2.CV_64F).var()
    return blur


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
        preds = round(image_feature_predict(full_name), 2)
        # Process your result for human
        result = str(preds)    # Convert to string
    return render_template('predict.html', image_file_name=file.filename, blur_value=result)


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/help', methods=['GET'])
def help():
    return render_template('help.html')


if __name__ == '__main__':
    app.run(debug=True)

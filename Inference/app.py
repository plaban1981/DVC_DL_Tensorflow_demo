import os
from flask import Flask, render_template, request
from flask import send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow. keras.models import load_model
from tensorflow.keras import backend as K
import logging
import argparse
import os
#
from src.utils.all_utils import read_yaml
from src.utils.models import load_full_model, get_unique_path_to_save_model
from src.utils.callbacks import get_callbacks
#
import numpy as np
import tensorflow as tf
#
import warnings
warnings.filterwarnings('ignore')
#
app = Flask(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
#
logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'running_logs.log'), level=logging.INFO, format=logging_str,
                    filemode="a")

#
UPLOAD_FOLDER = r'C:\Users\nayak\class_repository\DVC\DVC_DL_Tensorflow_demo\Inference\upload'
MODEL_FOLDER = r'C:\Users\nayak\class_repository\DVC\DVC_DL_Tensorflow_demo\artifacts\base_model\updated_VGG16_base_model.h5'

# Load And Prepare The Image
def load_image(filename):
    # load the image
    img = load_img(filename, target_size=(224, 224))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, 224, 224, 3)
    # center pixel data
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    return img

# load an image and predict the class
def predict(filename):
    # load the image
    img = load_img(filename, target_size=(224, 224))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, 224, 224, 3)
    # center pixel data
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    # load model
    model = load_model(MODEL_FOLDER )
    # predict the class
    result = model.predict(img)
    result = result.flatten()
    result = round(result[0])
    K.clear_session()
    return result


# home page
@app.route('/')
def home():
   return render_template('index.html')


# procesing uploaded file and predict it
@app.route('/upload', methods=['POST','GET'])
def upload_file():

    if request.method == 'GET':
        return render_template('index.html')
    else:
        file = request.files['image']
        full_name = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(full_name)

        indices = {0: 'Cat', 1: 'Dog'}
        result = predict(full_name)

        #accuracy = round(result[0][predicted_class] * 100, 2)
        label = indices[result]

    return render_template('predict.html', image_file_name = file.filename, label = label)


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':

    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")

    parsed_args = args.parse_args()

    try:
        logging.info(">>>>> Starting the web ui for cat dog classification ")
        app.run(host="0.0.0.0",port='8501',debug=False)
    except Exception as e:
        logging.exception(e)
        raise e
    
    


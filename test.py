from keras.preprocessing import image
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import cv2
import joblib
import base64
import json
import numpy as np

__class_name_to_number = {}
__class_number_to_name = []
__model = None


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name
    global __model

    with open("./artifact/class_dictionary.json", "r") as f:
        __class_number_to_name = json.load(f)

    if __model is None:
        __model = tf.keras.models.load_model('./artifact/saved_model/my_model')
    print("loading saved artifacts...done")


def get_cv2_image_from_base64_string(b64str):
    '''
    credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
    :param uri:
    :return:
    '''
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def get_cropped_image_if_2_eyes(image_base64_data, image_path):
    face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')
    cropped_faces = []
    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)
    return cropped_faces


def classify_image(image_base64_data, file_path=None):
    global re, batch_holder
    imgs = get_cropped_image_if_2_eyes(image_base64_data, file_path)
    IMG_SIZE = 256
    #     img = image.load_img(file_path, target_size=(IMG_SIZE,IMG_SIZE))
    for img in imgs:
        image = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        batch_holder = np.zeros((1, IMG_SIZE, IMG_SIZE, 3))
        batch_holder[0, :] = image
        re = __model.predict_classes(batch_holder)

    result = [{
        'class': __class_number_to_name[str(re[0])],
        'class_probability': np.around(__model.predict_proba(batch_holder) * 100, 2).tolist()[0],
        'class_dictionary': __class_number_to_name
    }]
    return result


if __name__ == '__main__':
    load_saved_artifacts()
    print(classify_image(None, "./test/crish.png"))

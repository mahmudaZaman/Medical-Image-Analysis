import io
import os

import boto3
import h5py
import s3fs
from PIL import ImageOps, Image
import numpy as np
from keras.models import load_model

from config import app_config


def upload_local_to_s3(tmp_local_path, bucket_name, s3_path):
    s3_client = boto3.client('s3')
    s3_client.upload_file(tmp_local_path, bucket_name, s3_path)
    os.remove(tmp_local_path)


def load_model_from_s3(tmp_local_path, bucket_name, s3_path):
    ignore_refresh = (os.path.isfile(tmp_local_path)) and (app_config.model.refresh is False)
    if ignore_refresh is True:
        print("using local model file")
    else:
        print("downloading model from s3")
        s3_client = boto3.client('s3')
        s3_client.download_file(bucket_name, s3_path, tmp_local_path)
    return load_model(tmp_local_path)


def classify(image, model, class_names):
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    prediction = model.predict(data)
    print("prediction value =============", prediction)
    index = 0 if prediction[0][0] <= 0.5 else 1
    class_name = class_names[index]
    return class_name


if __name__ == '__main__':
    model = load_model_from_s3("./data/interim/chest_xray.h5", app_config.storage.bucket_name,
                               app_config.storage.files.output_model_h5)

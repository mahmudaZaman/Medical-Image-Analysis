import os

import h5py
import s3fs
import streamlit as st
from PIL import Image

from config import app_config
from src.data.make_dataset import download_data
from src.models.train_model import run_train_pipeline
from util import classify, load_model_from_s3


def streamlit_run():
    st.title('Pneumonia classification')
    st.header('Please upload a chest X-ray image')
    file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

    model = load_model_from_s3("./data/interim/chest_xray.h5", app_config.storage.bucket_name,
                               app_config.storage.files.output_model_h5)
    class_names = ["NORMAL", "PNEUMONIA"]

    if file is not None:
        image = Image.open(file).convert('RGB')
        st.image(image, use_column_width=True)
        class_name = classify(image, model, class_names)
        st.write("## {}".format(class_name))


def model_run():
    run_train_pipeline()


if __name__ == '__main__':
    if app_config.data.refresh is True:
        print("downloading data from kaggle")
        download_data()
    mode = os.getenv("mode", "streamlit")
    print("mode", mode)
    if mode == "model":
        model_run()
    else:
        streamlit_run()

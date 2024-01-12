import streamlit as st
from keras.models import load_model
from PIL import Image
from util import classify

st.title('Pneumonia classification')
st.header('Please upload a chest X-ray image')
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
model = load_model('/Users/shuchi/Documents/work/personal/python/medical_image_analysis/chest_xray.h5')
class_names = ["NORMAL", "PNEUMONIA"]

if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)
    class_name = classify(image, model, class_names)
    st.write("## {}".format(class_name))

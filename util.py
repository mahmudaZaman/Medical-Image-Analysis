import io

import boto3
import h5py
from PIL import ImageOps, Image
import numpy as np
from s3fs import S3FileSystem


def save_model(model, bucket_name, object_key):
    # Create bytes io as target.
    with io.BytesIO() as model_io:
        # Instanciate h5 file using the io.
        with h5py.File(model_io, 'w') as model_h5:
            # Save the Keras model to h5 object (and indirectly to bytesio).
            model.save(model_h5)
            # Make sure the data is written entirely to the bytesio object.
            model_h5.flush()
        # Upload to S3.
        client = boto3.client('s3')
        client.put_object(
            Bucket=bucket_name,
            Key=object_key,
            Body=model_io,
        )


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

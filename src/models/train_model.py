import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.src.callbacks import EarlyStopping

from config import app_config
from src.features.data_ingestion import DataIngestion
from src.features.data_transformation import DataTransformation
from util import upload_local_to_s3


class ModelTrainer:
    def initiate_model_trainer(self, training_data, testing_data):
        IMAGE_SIZE = [224, 224]
        vgg_model = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
        vgg_model.trainable = False
        inputs = tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer")
        x = vgg_model(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
        x = tf.keras.layers.Dense(1024, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(1024, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="output_layer")(x)
        model = tf.keras.Model(inputs, outputs)

        # creating callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)

        # Compile the model
        model.compile(loss="binary_crossentropy",
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=["accuracy"])

        model.summary()
        model.fit(training_data, validation_data=testing_data, epochs=1,
                  steps_per_epoch=len(training_data), validation_steps=len(testing_data), callbacks=[early_stopping])

        tmp_local_path = './data/interim/chest_xray.h5'
        model.save(tmp_local_path)
        upload_local_to_s3(tmp_local_path, app_config.storage.bucket_name,
                           app_config.storage.files.output_model_h5)
        # save_model(model, app_config.storage.bucket_name, app_config.storage.files.output_model_h5)


def run_train_pipeline():
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    print(train_data_path, test_data_path)
    data_transformation = DataTransformation()
    training_data, testing_data = data_transformation.initiate_data_transformation(train_data_path,
                                                                                   test_data_path)
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(training_data, testing_data)
    print("model training completed")


if __name__ == '__main__':
    run_train_pipeline()

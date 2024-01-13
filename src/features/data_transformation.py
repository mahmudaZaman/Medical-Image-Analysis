from keras.preprocessing.image import ImageDataGenerator


class DataTransformation:
    @staticmethod
    def initiate_data_transformation(train_path, test_path):  # Todo verify
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True)

        test_datagen = ImageDataGenerator(rescale=1. / 255)
        training_data = train_datagen.flow_from_directory(train_path,
                                                          target_size=(224, 224),
                                                          batch_size=32,
                                                          class_mode='binary')
        testing_data = test_datagen.flow_from_directory(test_path,
                                                        target_size=(224, 224),
                                                        batch_size=32,
                                                        class_mode='binary')

        return (
            training_data, testing_data
        )

import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator


IMAGE_SIZE = [224, 224]
BATCH_SIZE = 32

path_train = '/Users/shuchi/Documents/work/personal/python/medical_image_analysis/dataset/train'
path_validation = '/Users/shuchi/Documents/work/personal/python/medical_image_analysis/dataset/val'
path_test = '/Users/shuchi/Documents/work/personal/python/medical_image_analysis/dataset/test'

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
training_data = train_datagen.flow_from_directory(path_train,
                                                 target_size = (224, 224),
                                                 batch_size = BATCH_SIZE,
                                                 class_mode = 'binary')
testing_data = test_datagen.flow_from_directory(path_test,
                                            target_size = (224, 224),
                                            batch_size = BATCH_SIZE,
                                            class_mode = 'binary')

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

# Compile the model
model.compile(loss="binary_crossentropy",
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

model.summary()
history = model.fit( training_data,validation_data=testing_data, epochs=5,
                     steps_per_epoch=len(training_data), validation_steps=len(testing_data))

model.save('chest_xray.h5')
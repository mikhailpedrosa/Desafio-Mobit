from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras import backend as K
from scipy.misc import imresize
import os
import numpy as np
import matplotlib.pyplot as plt

img_width, img_height = 150, 150
train_data_dir = 'C:/Users/Mikhail Pedrosa/PycharmProjects/mobit/questao3/data/train'
validation_data_dir = 'C:/Users/Mikhail Pedrosa/PycharmProjects/mobit/questao3/data/validation'
test_data_dir = 'C:/Users/Mikhail Pedrosa/PycharmProjects/mobit/questao3/data/test/'

classes = ["3", "4", "5", "indefinida"]
nb_train_samples = 115
nb_validation_samples = 47
epochs = 50
batch_size = 8

def create_model_cnn():

    if K.image_data_format() == 'channels_first':
         input_shape = (3, img_width, img_height)
    else:
         input_shape = (img_width, img_height, 3)

    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    return model

def train_model(model):

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    history2 = model.fit_generator(
        train_generator,
        steps_per_epoch=int(np.ceil(nb_train_samples) / float(batch_size)),
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

    # Loss Curves
    plt.figure(figsize=[8, 6])
    plt.plot(history2.history['loss'], 'r', linewidth=3.0)
    plt.plot(history2.history['val_loss'], 'b', linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves', fontsize=16)
    plt.savefig("loss.png")

    # Accuracy Curves
    plt.figure(figsize=[8, 6])
    plt.plot(history2.history['acc'], 'r', linewidth=3.0)
    plt.plot(history2.history['val_acc'], 'b', linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)
    plt.savefig("Accuracy.png")

    model.save_weights('cnn.h5')

if __name__ == '__main__':

    model = create_model_cnn()

    train_model(model)

    weights_path = 'cnn.h5'

    model.load_weights(weights_path)

    for file in os.listdir(test_data_dir):
        img = load_img(test_data_dir + file)
        img = imresize(img, (img_width, img_height))
        img = img_to_array(img)
        img = img.reshape(1, 150, 150, 3)

        print(test_data_dir + file, classes[int(model.predict_classes(img)[0])])

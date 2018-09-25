import sys
import os
from keras.models import *
import numpy as np
import keras
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
import  cv2
from keras.utils import np_utils

nb_classes=10
last_block_layer=249
img_height,img_width=139,139
batch_size=32
nb_epoch=20
#learn_rate=1e-4
#momentum=0.9
transformation_ratio=0.05
nb_images=50000
valid_images=2000

def load_cifar10_data(img_rows, img_cols):

    # Load cifar10 training and validation sets
    (X_train, Y_train), (X_valid, Y_valid) = cifar10.load_data()

    # Resize trainging images
    if K.image_dim_ordering() == 'th':
        X_train = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_train[:nb_images,:,:,:]])
        X_valid = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_valid[:valid_images,:,:,:]])
    else:
        X_train = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_train[:nb_images,:,:,:]])
        X_valid = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_valid[:valid_images,:,:,:]])

    # Transform targets to keras compatible format
    Y_train = np_utils.to_categorical(Y_train[:nb_images], nb_classes)
    Y_valid = np_utils.to_categorical(Y_valid[:valid_images], nb_classes)

    return X_train, Y_train, X_valid, Y_valid

def train(model_path):
    image_input = Input(shape=(img_width,img_height, 3))

    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Activation('relu')(x)
    predictions = Dense(nb_classes, activation='softmax')(x)

    model = Model(base_model.input,predictions)
    print(model.summary());

    for layer in model.layers[:last_block_layer]:
        layer.trainable = False
    for layer in model.layers[last_block_layer:]:
        layer.trainable = True
    
    rmsprop = keras.optimizers.RMSprop(lr=0.1, rho=0.9, epsilon=None, decay=0.0)
    model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])

    A_train,B_train,A_test,B_test = load_cifar10_data(139,139)
    np.save('A_train', A_train)
    np.save('B_train', B_train)
    np.save('A_test', A_test)
    np.save('B_test', B_test)
    print("Complete Loading")
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)
    datagen.fit(A_train)
    np.save("A_trainFinal",A_train)
    print("Start Tunning")
    model.fit_generator(datagen.flow(A_train,B_train,batch_size=batch_size), steps_per_epoch=nb_images / batch_size,
                        epochs=nb_epoch,
                        validation_data=(A_test,B_test))

    model_json = model.to_json()
    with open(os.path.join(os.path.abspath(model_path), 'modelRMSprorp.json'), 'w') as json_file:
        json_file.write(model_json)


model_path = ".\foldername\"
train(model_path)

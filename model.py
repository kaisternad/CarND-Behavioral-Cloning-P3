from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Dropout

import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg

origin_image_shape = (160, 320, 3)
img_shape = (64, 160, 3)
data_dir = "data/"


def read_driving_log(csv_file=data_dir + 'driving_log.csv'):
    return pd.read_csv(csv_file)


def read_image(filename):
    image = mpimg.imread(filename)
    return image


def flip(image, angle):
    image = np.array(image)
    flipped = cv2.flip(image, 1)
    angle = -angle
    return flipped, angle


def crop(image):
    image = image[66:130, 70:230]
    return image


def random_select_lcr(image_locations, augment):
    lrc_select = np.random.randint(3)
    # if augmentation is turned off, we always choose the center image
    if augment is False:
        lrc_select = 1

    angle = image_locations.get("steering")

    deps = {0: ("left", angle + 0.25),
            1: ("center", angle),
            2: ("right", angle - 0.25)
            }
    direction, angle_shift = deps[lrc_select]
    path_file = image_locations.get(direction).strip()
    return direction, angle_shift, path_file


def jitter(img):
    # see https://stackoverflow.com/questions/35152636/random-flipping-and-rgb-jittering-slight-value-change-of-image
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    rgb_shifted = np.dstack((
        np.roll(R, 10, axis=0),
        np.roll(G, 10, axis=1),
        np.roll(B, -10, axis=0)
    ))
    return rgb_shifted


def rand_brightness(img):
    # taken from https://github.com/vxy10/ImageAugmentation
    image1 = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    random_bright = .25 + np.random.uniform()
    image1[:, :, 2] = image1[:, :, 2] * random_bright
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1


def normalize(img):
    return (img - 128.) / 128.


def image_processing_step(sample, keep_probability=1, angle_threshold=0.01, augment=False):
    images = []
    angle_of_center_image = sample.get("steering")
    if abs(angle_of_center_image) <= angle_threshold and np.random.uniform() > keep_probability:
        return images

    # randomly select center, right or left image and adjust angle by (-) 0.25 if not center
    direction, angle, img_path = random_select_lcr(sample, augment)
    original_image = crop(read_image(data_dir + img_path))
    images.append((direction, angle, original_image))

    if augment is True:
        # flip image horizontally if angle > 0
        if np.random.uniform() >= 0.5 and abs(angle) > 0.0:
            flipped, flipped_angle = flip(original_image, angle)
            images.append(("flipped", flipped_angle, flipped))

        # add jitter
        if np.random.uniform() >= 0.5:
            jittered = jitter(original_image)
            images.append(("jittered", angle, jittered))

        # adjust random brightness
        if np.random.uniform() >= 0.5:
            random_b = rand_brightness(original_image)
            images.append(("brightness", angle, random_b))

    return images



def prepare_data(driving_log, augment=True):
    train, test = train_test_split(driving_log, test_size=0.2)

    samples = train.sample(frac=1)
    processed = []
    for index, row in samples.iterrows():
        pr = image_processing_step(row, augment=True)
        if len(pr) != 0:
            processed.extend(pr)

    direction, y_train, X_train = zip(*processed)
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    test_images_path = np.array(test["center"])
    X_test = []
    for i in range(0, len(test_images_path)):
        X_test.append(crop(read_image(data_dir + test_images_path[i])))

    X_test = np.array(X_test)
    y_test = np.array(test["steering"])
    return X_train, y_train, X_test, y_test


def build_NVIDIA_model():
    # NVIDIA Model, as per https://arxiv.org/pdf/1604.07316.pdf, but with ELU
    model = Sequential()
    model.add(Lambda(lambda x: ((x - 128.) / 128.), input_shape=img_shape))
    model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='elu', strides=(1, 1)))
    model.add(Conv2D(64, (3, 3), activation='elu', strides=(1, 1)))
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dropout(.5))
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(.2))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1, activation='tanh'))
    return model


driving_log = read_driving_log()
X_train, y_train, X_test, y_test = prepare_data(driving_log, augment=True)

model = build_NVIDIA_model()
model.summary()
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train,
            validation_data=(X_test, y_test),
            shuffle=True,
            epochs=8,
            verbose=1,
            batch_size=256)

model.save('model.h5')

import os
import numpy as np
import tensorflow as tf
import tflearn
from PIL import Image
from random import shuffle
from tqdm import tqdm
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.estimator import regression


LEARNING_RATE=1e-3
IMG_SIZE=50
IMG_DEPTH=3
WALLY_DIR='training/wally'
NOT_DIR='training/not'
MODEL_NAME='wanted_wally.model'


def build_model():
    tf.reset_default_graph()
    convnet = input_data(
        shape=[None, IMG_SIZE, IMG_SIZE, IMG_DEPTH], name='input')

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 128, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 256, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 128, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(
        convnet,
        optimizer='adam',
        learning_rate=LEARNING_RATE,
        loss='categorical_crossentropy',
        name='targets')

    return tflearn.DNN(convnet)


def package_dataset():
    wally_paths = [os.path.join(WALLY_DIR, f) for f in os.listdir(WALLY_DIR)]
    not_paths = [os.path.join(NOT_DIR, f) for f in os.listdir(NOT_DIR)]

    labelled_paths = \
        zip(wally_paths, [np.array([0, 1])]*len(wally_paths)) + \
        zip(not_paths, [np.array([1, 0])]*len(wally_paths))

    training_data = []
    for path, label in tqdm(labelled_paths):
        img = Image.open(path)
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
        img = np.array(img)
        img = img[:, :, ::-1].copy()
        training_data.append([img, label])
    shuffle(training_data)
    return training_data


def run_training(training_data):
    model = build_model()
    train = training_data[:-500]
    test = training_data[-500:]

    X = np.array([i[0] for i in train]) \
          .reshape(-1, IMG_SIZE, IMG_SIZE, IMG_DEPTH)
    Y = [i[1] for i in train]

    test_x = np.array([i[0] for i in test]) \
               .reshape(-1, IMG_SIZE, IMG_SIZE, IMG_DEPTH)
    test_y = [i[1] for i in test]

    model.fit(
        X_inputs={ 'input': X }, Y_targets={ 'targets': Y }, \
        validation_set=({ 'input': test_x }, { 'targets': test_y }), \
        n_epoch=3, snapshot_step=250, show_metric=True, run_id=MODEL_NAME)
    model.save(MODEL_NAME)


def run_hypothesis(filepath,
                   crop_size=75,
                   crop_overlap=50,
                   threshold=0.999):
    model = build_model()
    model.load(MODEL_NAME)

    image = Image.open(filepath)
    (image_w, image_h) = image.size
    highlighted = Image.new('RGB', image.size, (0, ) * 3)
    highlighted.paste(image.convert('L'))

    crop_increment = crop_size - crop_overlap
    horizontal_crops = int(image_w / crop_increment)
    vertical_crops = int(image_h / crop_increment)

    for i in tqdm(xrange(horizontal_crops * vertical_crops)):
        x1 = int(i % horizontal_crops) * crop_increment
        y1 = int(i / horizontal_crops) * crop_increment
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        crop = image.crop((x1, y1, x2, y2))
        data = crop.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
        data = np.array(data)
        data = data[:, :, ::-1].copy()

        prediction = model.predict([data])
        if (prediction[0][1] >= threshold):
            highlighted.paste(crop, (x1, y1))
    return highlighted

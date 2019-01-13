import numpy as np
from sklearn.preprocessing import LabelBinarizer
import pickle
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import scipy.ndimage as nd
from keras.models import Sequential,load_model
from keras import regularizers
from keras import initializers
from keras import metrics
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Convolution2D, MaxPooling2D, BatchNormalization, ReLU, LeakyReLU
from keras.callbacks import LearningRateScheduler
from keras import models, layers
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 8} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)

def unpickle(file):
    
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
        
    encoder = LabelBinarizer()
    data['labels'] = encoder.fit_transform(data['labels'])

    return data


def get_data():
    images, labels = [], []
    test_img, test_labels = [], []
    for i in range(1, 6):
        batch = unpickle('cifar-10-batches-py/data_batch_{}'.format(str(i)))
        images.append(batch['data'])
        labels.append(batch['labels'])
    idx = np.arange(50000)
    np.random.shuffle(idx)
    images, labels = np.concatenate(images, axis = 0)[idx], np.concatenate(labels, axis = 0)[idx]
    
    test_images = unpickle('cifar-10-batches-py/test_batch')['data']
    test_labels = unpickle('cifar-10-batches-py/test_batch')['labels']

    images = np.array([[image[:1024], image[1024:2048], image[2048:3072]] for image in images])
    test_images = np.array([[image[:1024], image[1024:2048], image[2048:3072]] for image in test_images])
    
    images = images.reshape((-1, 32, 32, 3))
    test_images = test_images.reshape((-1, 32, 32, 3))
    
    train_images, val_images = images[:45000], images[45000:]
    train_labels, val_labels = labels[:45000], labels[45000:]
    
    return train_images, train_labels, val_images, val_labels, test_images, test_labels


def write_results(model, comments, train_history, filename):
    file = open('results/' + filename, 'w')
    acc, loss, val_acc, val_loss = train_history['acc'], train_history['loss'], train_history['val_acc'], train_history['val_loss']
    
    model.summary(print_fn=lambda x: file.write(x + '\n'))
    file.write(comments + '\n')

    for metric in acc, loss, val_acc, val_loss:
        file.write('*')
        for value in metric:
            file.write(str(value) + ' ')
        file.write('\n')
    file.close()

def cnn_1(dropout = 0.0, l2 = 0.0):
    model = Sequential()
    init = initializers.glorot_normal(seed=0)

    model.add(Convolution2D(64, (5, 5), padding='same', strides=(2,2), input_shape=(32, 32, 3), kernel_initializer=init))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(128, (5, 5), padding='same', kernel_initializer=init))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(256, (3, 3), padding='same', kernel_initializer=init))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(LeakyReLU())
    if dropout > 0: model.add(Dropout(dropout))
    model.add(Dense(512))
    model.add(LeakyReLU())
    if dropout > 0: model.add(Dropout(dropout))
    model.add(Dense(10, activation='softmax'))

    return model

def cnn_2(dropout = 0.0, l2 = 0.0):
    model = Sequential()
    init = initializers.glorot_normal(seed=0)

    model.add(Convolution2D(64, (5, 5), padding='same', strides=(2,2), input_shape=(32, 32, 3), kernel_initializer=init))
    model.add(LeakyReLU())
    model.add(Convolution2D(64, (5, 5), padding='same', kernel_initializer=init))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(128, (5, 5), padding='same', kernel_initializer=init))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Convolution2D(128, (5, 5), padding='same', kernel_initializer=init))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(256, (3, 3), padding='same', kernel_initializer=init))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Convolution2D(256, (3, 3), padding='same', kernel_initializer=init))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))


    model.add(Flatten())
    model.add(Dense(1024))
    model.add(LeakyReLU())
    if dropout > 0: model.add(Dropout(dropout))
    model.add(Dense(512))
    model.add(LeakyReLU())
    if dropout > 0: model.add(Dropout(dropout))
    model.add(Dense(10, activation='softmax'))

    return model


def cnn_3(dropout = 0.0, l2 = 0.0):
    model = Sequential()
    init = initializers.glorot_normal(seed=0)

    model.add(Convolution2D(64, (5, 5), padding='same', strides=(2,2), input_shape=(32, 32, 3), kernel_initializer=init))
    model.add(LeakyReLU())
    model.add(Convolution2D(64, (5, 5), padding='same', kernel_initializer=init))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(128, (5, 5), padding='same', kernel_initializer=init))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Convolution2D(128, (5, 5), padding='same', kernel_initializer=init))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Convolution2D(128, (5, 5), padding='same', kernel_initializer=init))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(256, (3, 3), padding='same', kernel_initializer=init))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Convolution2D(256, (3, 3), padding='same', kernel_initializer=init))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Convolution2D(256, (3, 3), padding='same', kernel_initializer=init))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))


    model.add(Flatten())
    model.add(Dense(1024))
    model.add(LeakyReLU())
    if dropout > 0: model.add(Dropout(dropout))
    model.add(Dense(512))
    model.add(LeakyReLU())
    if dropout > 0: model.add(Dropout(dropout))
    model.add(Dense(10, activation='softmax'))

    return model

def cnn_4(dropout = 0.0, l2 = 0.0):
    model = Sequential()
    init = initializers.glorot_normal(seed=0)

    model.add(Convolution2D(128, (5, 5), padding='same', strides=(2,2), input_shape=(32, 32, 3), kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Convolution2D(128, (5, 5), padding='same', kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(256, (5, 5), padding='same', kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Convolution2D(256, (5, 5), padding='same', kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Convolution2D(256, (5, 5), padding='same', kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(512, (3, 3), padding='same', kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Convolution2D(512, (3, 3), padding='same', kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Convolution2D(512, (3, 3), padding='same', kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2,2)))


    model.add(Flatten())
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    if dropout > 0: model.add(Dropout(dropout))
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    if dropout > 0: model.add(Dropout(dropout))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax'))

    return model

def residual_block(y, out_channels, kernel_size, l2):
    if not(y.shape[-1] == out_channels):
        shortcut = layers.Conv2D(out_channels, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
    else:
        shortcut = y

    y = layers.Conv2D(out_channels, kernel_size=kernel_size, strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(l2))(y)
    y = layers.BatchNormalization()(y)
    y = LeakyReLU()(y)

    y = layers.Conv2D(out_channels, kernel_size=kernel_size, strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(l2))(y)
    y = layers.add([shortcut, y])
    y = layers.BatchNormalization()(y)
    y = LeakyReLU()(y)

    return y

def resnet_1(x, l2 = 0.0):
    x = layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(l2))(x)

    x = residual_block(x, 16, (3, 3), l2)
    x = residual_block(x, 16, (3, 3), l2)    
    x = residual_block(x, 16, (3, 3), l2)
    x = layers.MaxPool2D(pool_size=(2, 2), padding='same')(x)

    x = residual_block(x, 32, (3, 3), l2)
    x = residual_block(x, 32, (3, 3), l2)
    x = residual_block(x, 32, (3, 3), l2)
    x = layers.MaxPool2D(pool_size=(2, 2), padding='same')(x)
    
    x = residual_block(x, 64, (3, 3), l2)
    x = residual_block(x, 64, (3, 3), l2)
    x = residual_block(x, 64, (3, 3), l2)
    x = layers.MaxPool2D(pool_size=(2, 2), padding='same')(x)

    
    x = layers.Flatten()(x)
    x = layers.Dense(1000, kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = layers.Dense(10, activation = 'softmax')(x)

    return x


def schedule(step):
    return 10**-2 * 10**-(step//25)

train_data, train_labels, val_data, val_labels, test_data, test_labels = get_data()
datagen = ImageDataGenerator(samplewise_center = True, samplewise_std_normalization = True, width_shift_range = 0.1, height_shift_range = 0.1, horizontal_flip = True, fill_mode = 'constant')
testgen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)

datagen.fit(train_data)
datagen = datagen.flow(train_data, train_labels, batch_size=32)

testgen.fit(val_data)
testgen = testgen.flow(val_data, val_labels, batch_size=32)

image_tensor = layers.Input(shape=(32, 32, 3))
network_output = resnet_1(image_tensor, l2 = 0.0001)
model = models.Model(inputs=[image_tensor], outputs=[network_output])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', metrics.top_k_categorical_accuracy])
model.summary()

train_history = {'acc':[], 'loss':[], 'top_k_categorical_accuracy': [], 'val_acc':[], 'val_loss':[],  'val_top_k_categorical_accuracy': []}

lrate = LearningRateScheduler(schedule)
for i in range(101):
    if i % 5 == 0:
        model.save('results/resnet_1_A_L-4.h5')
    train_history_ = model.fit_generator(datagen, steps_per_epoch=len(train_data)/32, epochs=1,
                                             validation_data = testgen, validation_steps = len(val_data)/32, callbacks=[lrate])
    train_history['acc'].append(train_history_.history['acc'][0])
    train_history['loss'].append(train_history_.history['loss'][0])
    train_history['top_k_categorical_accuracy'].append(train_history_.history['top_k_categorical_accuracy'][0])
    train_history['val_acc'].append(train_history_.history['val_acc'][0])
    train_history['val_loss'].append(train_history_.history['val_loss'][0])
    train_history['val_top_k_categorical_accuracy'].append(train_history_.history['val_top_k_categorical_accuracy'][0])


write_results(model,  "L = 10**-4 everywhere, with augmentation", train_history, 'resnet_1A_L-4.txt')

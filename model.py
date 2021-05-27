import numpy as np
import pandas as pd

from keras import regularizers, activations
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.utils import np_utils, to_categorical

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
from matplotlib import pyplot as plt


# load the preprocessed datat
us8k_df = pd.read_pickle("us8k_df.pkl")


def init_data_aug():
    train_datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        fill_mode='constant',
        cval=-80.0,
        width_shift_range=0.1,
        height_shift_range=0.0)

    val_datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        fill_mode='constant',
        cval=-80.0)

    return train_datagen, val_datagen


def init_model():
    model1 = Sequential()

    # layer-1
    model1.add(Conv2D(filters=24, kernel_size=5, input_shape=(128, 128, 1),
                      kernel_regularizer=regularizers.l2(1e-3)))
    model1.add(MaxPooling2D(pool_size=(3, 3), strides=3))
    model1.add(Activation(activations.relu))

    # layer-2
    model1.add(Conv2D(filters=36, kernel_size=4, padding='valid', kernel_regularizer=regularizers.l2(1e-3)))
    model1.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model1.add(Activation(activations.relu))

    # layer-3
    model1.add(Conv2D(filters=48, kernel_size=3, padding='valid'))
    model1.add(Activation(activations.relu))

    model1.add(GlobalAveragePooling2D())

    # layer-4 (1st dense layer)
    model1.add(Dense(60, activation='relu'))
    model1.add(Dropout(0.5))

    # layer-5 (2nd dense layer)
    model1.add(Dense(10, activation='softmax'))

    # compile
    model1.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    return model1


model = init_model()
print(model.summary())




def train_test_split(fold_k, data, X_dim=(128, 128, 1)):
    X_train = np.stack(data[data.fold != fold_k].melspectrogram.to_numpy())
    X_test = np.stack(data[data.fold == fold_k].melspectrogram.to_numpy())

    y_train = data[data.fold != fold_k].label.to_numpy()
    y_test = data[data.fold == fold_k].label.to_numpy()

    XX_train = X_train.reshape(X_train.shape[0], *X_dim)
    XX_test = X_test.reshape(X_test.shape[0], *X_dim)

    yy_train = to_categorical(y_train)
    yy_test = to_categorical(y_test)

    return XX_train, XX_test, yy_train, yy_test


def process_fold(fold_k, data, epochs=100, num_batch_size=32):
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(fold_k, data)

    # init data augmention
    train_datagen, val_datagen = init_data_aug()

    # fit augmentation
    train_datagen.fit(X_train)
    val_datagen.fit(X_train)

    # init model
    model = init_model()

    # pre-training accuracy
    score = model.evaluate(val_datagen.flow(X_test, y_test, batch_size=num_batch_size), verbose=0)
    print("Pre-training accuracy: %.4f%%\n" % (100 * score[1]))

    # train the model
    start = datetime.now()
    history = model.fit(train_datagen.flow(X_train, y_train, batch_size=num_batch_size),
                        steps_per_epoch=len(X_train) / num_batch_size,
                        epochs=epochs,
                        validation_data=val_datagen.flow(X_test, y_test, batch_size=num_batch_size))
    end = datetime.now()
    print("Training completed in time: ", end - start, '\n')

    return history


def show_results(tot_history):
    """Show accuracy and loss graphs for train and test sets."""

    for i, history in enumerate(tot_history):
        print('\n({})'.format(i + 1))

        plt.figure(figsize=(15, 5))

        plt.subplot(121)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.grid(linestyle='--')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')

        plt.subplot(122)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.grid(linestyle='--')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')

        plt.show()

        print('\tMax validation accuracy: %.4f %%' % (np.max(history.history['val_accuracy']) * 100))
        print('\tMin validation loss: %.5f' % np.min(history.history['val_loss']))


FOLD_K = 1
REPEAT = 1

history1 = []

# for i in range(REPEAT):
#     print('-' * 80)
#     print("\n({})\n".format(i + 1))
#
#     history = process_fold(FOLD_K, us8k_df, epochs=100)
#     history1.append(history)
#
# show_results(history1)
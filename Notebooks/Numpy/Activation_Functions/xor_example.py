import os
import shutil
import argparse

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import matplotlib.pyplot as plt
import imageio


parser = argparse.ArgumentParser()
parser.add_argument('-a', '--activation', help='activation function from keras', default='linear')
args = parser.parse_args()


ITERATIONS = 1000 
REPEAT = 5
INTERVAL = 5
DIRECTORY = 'xor_imgs'
ANGLE = 0
ACTIVATION = args.activation


def make_net(activation='linear'):
    inp = tf.keras.Input(shape=(2,))
    x = tf.keras.layers.Dense(4, activation=activation)(inp)
    out = tf.keras.layers.Dense(1)(x)

    return tf.keras.Model(inputs=inp, outputs=out), Adam(lr=0.01)


def clear_dir(directory):
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        os.unlink(filepath)


def get_ims(directory, iterations):
    filenames = os.listdir(directory)
    for i in range(iterations):
        filename = f'img_{i}.png'
        if filename in filenames:
            yield os.path.join(directory, filename)


if __name__ == '__main__':
    fig = plt.figure(figsize=(5, 5), dpi=200)
    ax = fig.add_subplot(111, projection='3d') 

    try:
        os.mkdir(DIRECTORY)
    except OSError:
        pass

    clear_dir(DIRECTORY)

    net, opt = make_net(ACTIVATION)
    loss_fn = MeanSquaredError()

    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    y = np.array([[0.0], [1.0], [1.0], [0.0]])

    previous_loss = np.inf
    for i in range(ITERATIONS):
        with tf.GradientTape() as tape:
            y_hat = net(X)
            loss = loss_fn(y, y_hat)
            if i % 100 == 0:
                print(loss)

        if i % INTERVAL == 0:
            ax.clear()

            ax.scatter(X[:, 0], X[:, 1], y, label='true', depthshade=False)
            ax.scatter(X[:, 0], X[:, 1], y_hat, color='r', label='model', depthshade=False)
            ax.view_init(elev=5., azim=ANGLE)

            ANGLE += 1
            ANGLE %= 360

        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper left')
            fig.suptitle(f'{"Linear" if ACTIVATION == "linear" else "Nonlinear"} NN vs XOR')

            plt.tight_layout()

        if i % INTERVAL == 0:
            plt.savefig(os.path.join(DIRECTORY, f'img_{i}.png'))

        if ACTIVATION == 'linear':
            if abs(previous_loss - loss) < 1.0e-10:
                break
        else:
            if loss < 1.0e-5:
                break

        previous_loss = loss

        gradients = tape.gradient(loss, net.trainable_variables)
        opt.apply_gradients(zip(gradients, net.trainable_variables))

    imgs = []
    for filename in get_ims(DIRECTORY, ITERATIONS):
        imgs.append(imageio.imread(filename))
    
    for _ in range(REPEAT):
        imgs.append(imageio.imread(filename))

    imageio.mimsave(f'gifs/nn_xor_{ACTIVATION}.gif', imgs, duration=.1)
    shutil.rmtree('xor_imgs')

import os
import shutil

import imageio
import numpy as np
import matplotlib.pyplot as plt


ITERATIONS = 10000
REPEAT = 5
INTERVAL = 5
MODELS = {}
DIRECTORY = 'single_imgs'


class UnivariateOLS:
    def __init__(self, lr=0.01):
        self.beta = np.random.randn()
        self.bias = np.random.randn()
        self.lr = lr

    def __call__(self, x):
        return x * self.beta + self.bias

    def cost(self, x, y):
        return np.square(y - self(x)).mean()

    def update(self, x, y):
        d_beta, d_bias = self.__derive__(x, y)

        self.beta += d_beta * -self.lr
        self.bias += d_bias * -self.lr

    def __derive__(self, x, y):
        y_hat = self(x)
        d_beta = (-2 / len(x)) * (x * (y - y_hat)).sum()
        d_bias = (-2 / len(x)) * (y - y_hat).sum()
        return d_beta, d_bias


class RandomModel:
    def __init__(self):
        n = np.random.randint(1, 6)

        self.betas = np.random.uniform(-3, 3, size=(n,))
        self.biases = np.random.uniform(-3, 3, size=(n,))

    def __call__(self, x):
        for beta, bias in zip(self.betas, self.biases):
            x = x * beta + bias
        return x


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
    fig, axs = plt.subplots(3, 3, figsize=(5, 5), dpi=200)

    try:
        os.mkdir(DIRECTORY)
    except OSError:
        pass

    clear_dir(DIRECTORY)

    for i in range(ITERATIONS):
        costs = []
        for x in range(3):
            for y in range(3):
                try:
                    rm, ols = MODELS[(x, y)]
                except KeyError:
                    rm = RandomModel()
                    ols = UnivariateOLS()

                    MODELS[(x, y)] = (rm, ols)

                if i % INTERVAL == 0:
                    vals = np.arange(10)
                    
                    ax = axs[x, y]
                    ax.clear()

                    ax.plot(vals, rm(vals), label='true')
                    ax.plot(vals, ols(vals), color='r', label='model')

                X_true = np.random.normal(size=(32,))
                y_true = rm(X_true)

                ols.update(X_true, y_true)
                costs.append(ols.cost(X_true, y_true))

        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper left')
            fig.suptitle('Random Linear Models')

            plt.tight_layout()

        if i % INTERVAL == 0:
            plt.savefig(os.path.join(DIRECTORY, f'img_{i}.png'))

        if max(costs) < 1.0e-2:
            break

    imgs = []
    for filename in get_ims(DIRECTORY, ITERATIONS):
        imgs.append(imageio.imread(filename))
    
    for _ in range(REPEAT):
        imgs.append(imageio.imread(filename))

    imageio.mimsave('gifs/single_var.gif', imgs, duration=.1)
    shutil.rmtree('single_imgs')

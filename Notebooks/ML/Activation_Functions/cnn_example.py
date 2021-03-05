import tensorflow as tf
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


INPUT_DIM = (np.random.randint(1, 33), np.random.randint(1, 33), 3)
NUM_LAYERS = np.random.randint(1, 4)
OUTPUT_DIM = np.random.randint(1, 129)


def random_model():
    inp = tf.keras.Input(shape=INPUT_DIM)

    for layer in range(NUM_LAYERS):
        if layer == 0:
            x = tf.keras.layers.Conv2D(
                np.random.randint(1, 33),
                (3, 3),
                bias_initializer='glorot_uniform',
                activation='linear'
            )(inp)
        else:
            x = tf.keras.layers.Conv2D(
                np.random.randint(1, 33),
                (3, 3),
                bias_initializer='glorot_uniform',
                activation='linear'
            )(x)
        
        if np.random.uniform() > .5:
            x = tf.keras.layers.AveragePooling2D()(x)

    x = tf.keras.layers.Flatten()(x)

    out = tf.keras.layers.Dense(OUTPUT_DIM, bias_initializer='glorot_uniform')(x)

    return tf.keras.Model(inputs=inp, outputs=out)


if __name__ == '__main__':
    model = random_model()
    model.summary()

    X = np.random.uniform(size=(8192, *INPUT_DIM))
    y = model(X)

    linear_model = LinearRegression().fit(X.reshape(8192, -1), y)

    X_test = np.random.uniform(size=(1024, *INPUT_DIM))
    y_true = model(X_test)
    y_pred = linear_model.predict(X_test.reshape(1024, -1))

    print(f'r2 score: {r2_score(y_true, y_pred)}')
    print(f'mse: {mean_squared_error(y_true, y_pred)}')

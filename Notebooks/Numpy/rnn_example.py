import tensorflow as tf
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


INPUT_DIM = (100, 5)
NUM_LAYERS = np.random.randint(1, 4)
OUTPUT_DIM = np.random.randint(1, 129)


def random_model():
    inp = tf.keras.Input(shape=INPUT_DIM)

    for layer in range(NUM_LAYERS):
        if layer+1 == NUM_LAYERS:
            l = tf.keras.layers.SimpleRNN(
                np.random.randint(1, 129),
                bias_initializer='glorot_uniform',
                activation='linear'
            )
        else:
            l = tf.keras.layers.SimpleRNN(
                np.random.randint(1, 129),
                return_sequences=True,
                bias_initializer='glorot_uniform',
                activation='linear'
            )

        if layer == 0:
            x = l(inp)
        else:
            x = l(x)

    out = tf.keras.layers.Dense(OUTPUT_DIM, bias_initializer='glorot_uniform')(x)

    return tf.keras.Model(inputs=inp, outputs=out)


if __name__ == '__main__':
    model = random_model()
    model.summary()

    X = np.random.uniform(size=(1024, *INPUT_DIM))
    y = model(X)

    linear_model = LinearRegression().fit(X.reshape(1024, -1), y)

    X_test = np.random.uniform(size=(1024, *INPUT_DIM))
    y_true = model(X_test)
    y_pred = linear_model.predict(X_test.reshape(1024, -1))

    print(f'r2 score: {r2_score(y_true, y_pred)}')
    print(f'mse: {mean_squared_error(y_true, y_pred)}')

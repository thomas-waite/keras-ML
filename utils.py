from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from typing import Tuple

MODEL_PATH: str = './weights/model'


def generate_plot(history, model_var: str):
    plt.plot(history.history[model_var])
    plt.plot(history.history[f'{model_var}'])
    plt.title('model loss')
    plt.ylabel(model_var)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def load_data() -> Tuple[list, list, int]:
    dataset = pd.read_csv('pima-indians-diabetes.csv')
    raw_data = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')

    # split into input and output variables
    input_dim = 8
    X = raw_data[:, 0:input_dim]
    y = raw_data[:, input_dim]

    return X, y, input_dim


def load_keras_model() -> Any:
    return load_model(MODEL_PATH)

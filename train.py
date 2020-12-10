from analysis import generate_plot
from model import model_factory
import numpy as np
import pandas as pd
from keras.models import load_model

MODEL_PATH = './weights/model'


def load_data():
    dataset = pd.read_csv('pima-indians-diabetes.csv')
    raw_data = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')

    # split into input and output variables
    input_dim = 8
    X = raw_data[:, 0:input_dim]
    y = raw_data[:, input_dim]

    return X, y, input_dim


def train(data, labels, model):
    history = model.fit(data, labels, epochs=150, batch_size=10)
    generate_plot(history, 'loss')
    model.save(MODEL_PATH)


def load_model():
    return load_model(MODEL_PATH)


X, y, input_dim = load_data()
model = model_factory(input_dim)
train(X, y, model)

from utils import generate_plot, load_data, MODEL_PATH
from model import model_factory
from keras.models import load_model


def train(data, labels, model):
    history = model.fit(data, labels, epochs=150, batch_size=10)
    generate_plot(history, 'loss')
    model.save(MODEL_PATH)


X, y, input_dim = load_data()
model = model_factory(input_dim)
train(X, y, model)

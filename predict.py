from utils import load_data, load_keras_model


def predict(data):
    model = load_keras_model()
    predictions = model.predict_classes(data)
    for i in range(5):
        print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))


X, y, _ = load_data()
predict(X)

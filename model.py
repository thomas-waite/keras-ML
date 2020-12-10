from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model


def model_factory(input_dim: int) -> Sequential:
    model = Sequential()
    model.add(Dense(12, input_dim=input_dim, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    plot_model(model, to_file='model_plot.png',
               show_shapes=True, show_layer_names=True)
    return model

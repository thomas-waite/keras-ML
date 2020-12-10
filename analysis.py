import matplotlib.pyplot as plt


def generate_plot(history, model_var: str):
    plt.plot(history.history[model_var])
    plt.plot(history.history[f'{model_var}'])
    plt.title('model loss')
    plt.ylabel(model_var)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

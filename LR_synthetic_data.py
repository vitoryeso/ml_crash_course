import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

print(tf.__version__)

# lets define two functions
# build_model(learning_rate)
# train_model(model, feature, label, epochs)
# plot_the_model(trained_weight, trained_bias, features, labels)

def build_model(learning_rate):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(1, input_shape=(1,)))
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate),
            loss="mean_squared_error",
            metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

def train_model(model, feature, label, epochs, batch_size):
    history = model.fit(feature, label, epochs=epochs, batch_size=batch_size)

    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]

    hist = pd.DataFrame(history.history)
    epochs = history.epoch # return a vector with the epochs

    rmse = hist["root_mean_squared_error"]

    return trained_weight, trained_bias, epochs, rmse

def plot_the_model(trained_weight, trained_bias, features, labels):
    plt.xlabel("feature")
    plt.ylabel("label")

    plt.scatter(features, labels)

    # create a red line indicating the model. this line inits in Po and ends in P1
    x0 = 0
    y0 = trained_bias

    x1 = features[-1]
    y1 = (x1 * trained_weight) + trained_bias
    plt.plot([x0, x1], [y0, y1], c="r")

    plt.show()

def plot_the_loss_curve(epochs, rmse):
   plt.plot(epochs, rmse, label="loss")
   plt.xlabel("epoch")
   plt.ylabel("root mean squared error")

   plt.show()

my_feature = ([1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0, 10.0, 11.0, 12.0])
my_label   = ([5.0, 8.8,  9.6, 14.2, 18.8, 19.5, 21.4, 26.8, 28.9, 32.0, 33.8, 38.2])

learning_rate = 0.01
epochs = 500
batch_size = 12

model = build_model(learning_rate)
trained_weight, trained_bias, epochs, rmse = train_model(model, my_feature, my_label, epochs, batch_size)
plot_the_model(trained_weight, trained_bias, my_feature, my_label)
plot_the_loss_curve(epochs, rmse)




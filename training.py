from keras.datasets import mnist
from keras.models import Model
from keras import layers
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt


def model():
    input_layer = layers.Input(shape=(28, 28, 1))
    x = layers.Conv2D(
        filters=32,
        strides=1,
        kernel_size=3,
        padding='same',
        use_bias=False
    )(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)

    x = layers.Conv2D(
        filters=32,
        strides=1,
        kernel_size=3,
        padding="same",
        use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(units=10)(x)
    output_layer = layers.Softmax()(x)
    return Model(input_layer, output_layer)


def ffnn():
    input_layer = layers.Input(shape=(28, 28, 1))
    x = layers.Flatten()(input_layer)
    x = layers.Dense(units=64, use_bias=False, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(rate=0.2)(x)
    x = layers.Dense(units=64, use_bias=False, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(rate=0.2)(x)
    x = layers.Dense(units=128, use_bias=False, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(rate=0.2)(x)
    x = layers.Dense(units=128, use_bias=False, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(rate=0.2)(x)
    output_layer = layers.Dense(units=10, use_bias=False, activation="softmax")(x)
    return Model(input_layer, output_layer)


def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train, x_test = x_train / 255.0, x_test / 255.0

    cnn_model = model()

    cnn_model.summary()
    cnn_model.compile(
        loss=SparseCategoricalCrossentropy(),
        optimizer=Adam(),
        metrics=["accuracy"]
    )
    epochs = 50
    history = cnn_model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        shuffle=True,
        batch_size=32,
    )

    plt.figure(figsize=(5, 5))
    plt.subplot(1, 2, 1)
    plt.plot([i for i in range(epochs)], history.history["accuracy"], label="accuracy")
    plt.plot([i for i in range(epochs)], history.history["val_accuracy"], label="val_accuracy")
    plt.subplot(1, 2, 2)
    plt.plot([i for i in range(epochs)], history.history["loss"], label="loss")
    plt.plot([i for i in range(epochs)], history.history["val_loss"], label="val_loss")
    plt.legend()
    plt.show()

    cnn_model.save("models/model.h5")


if __name__ == "__main__":
    main()

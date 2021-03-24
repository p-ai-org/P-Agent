import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
import matplotlib.pyplot as plt

import os
import pdb

class package_predict():
    def __init__(self, test_size=0.2):
        self.test_size = test_size

        self.image_scale = 255
        self.label_scale = None

        self.dirpath = os.path.dirname(os.path.realpath(__file__))

    def load_data_set(self):
        dataset = pickle.load(open(os.path.join("..", "data", "package_predict",
                "images_labels.pkl"), "rb"))
        images = dataset["data"]
        labels = dataset["labels"]

        self.label_scale = np.abs(labels).max(axis=0)
        labels = labels / self.label_scale
        images = images / self.image_scale

        indices = np.random.permutation(images.shape[0])
        split_i = int(self.test_size * len(indices))
        test_idx, train_idx = indices[:split_i], indices[split_i:]

        train_ds, test_ds = images[train_idx], images[test_idx]
        train_labels, test_labels = labels[train_idx], labels[test_idx]

        return train_ds, train_labels, test_ds, test_labels
            
    def train_model(self, save_model=True, epochs=10):
        # Load dataset
        train_ds, train_labels, test_ds, test_labels = self.load_data_set()

        # Conv2D layers
        model = tf.keras.models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=train_ds.shape[1:]))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation="relu"))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation="relu"))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation="relu"))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation="relu"))

        # Dense layers
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dense(train_labels.shape[1]))

        # Compile and train model
        model.compile(optimizer="adam",
                loss=tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error"),
                metrics=["mean_squared_error"])
        history = model.fit(train_ds, train_labels, epochs=epochs,
                validation_data=(test_ds, test_labels))

        self.model = model
        self.history = history

        # Save model 
        if save_model:
            model.save("package_predict_model") # Alternatively can save as an H5 for computational
                                                # savings. Test with wholesale first
            model_params = {"image_scale": self.image_scale, "label_scale": self.label_scale}
            pickle.dump(model_params, open("package_predict_model_params.pkl", "wb"))

        # Evaluate model
        plt.plot(history.history["mean_squared_error"], label="mean_squared_error")
        # plt.plot(history.history["val_accuracy"], label="val_accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        # plt.ylim([0,5, 1])
        plt.legend(loc="lower right")
        plt.show()

        test_loss, test_mse = model.evaluate(test_ds, test_labels, verbose=2)
        print(test_mse)

    def predict(self, img):
        """
        Args:
        img = input png image from drone

        Output:
        rel_pos = relative position
        pitch = relative pitch
        yaw = relative yaw
        roll = relative roll
        """
        K.clear_session()
        self.model = tf.keras.models.load_model(os.path.join(self.dirpath, "package_predict_model"))
        self.model._make_predict_function()

        if self.label_scale is None:
            model_params = pickle.load(open(os.path.join(self.dirpath,
                    "package_predict_model_params.pkl"), "rb"))
            self.image_scale = model_params["image_scale"]
            self.label_scale = model_params["label_scale"]

        prediction = self.model.predict(img / self.image_scale)
        return prediction * self.label_scale
        K.clear_session()

if __name__ == "__main__":
    p = package_predict()

    dataset = pickle.load(open(os.path.join("..", "data", "package_predict",
            "images_labels.pkl"), "rb"))
    images = dataset["data"]
    labels = dataset["labels"]
    prediction = p.predict(np.expand_dims(images[0], axis=0))

#     mse = np.zeros(1000)
#     for i in range(1, 1001):
#         prediction = p.predict(np.expand_dims(images[i], axis=0))
#         pdb.set_trace()
#         mse[i] = np.mean((prediction - labels[i]) ** 2)
#         print(f"i {i}: mse {mse[i]}")

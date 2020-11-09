import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


class package_predict():
    def __init__(image, store_vals):

        # Image Parameters
        self.store_vals = store_vals
        self.img_height = 240
        self.img_width = 320

        self.batch_size = 128

        # Network Parameters
        # num_input = img_height*img_width # MNIST data input (img shape: 28*28)
        self.num_classes = 4 # Output rel. distance, pose (3)

        # tf Graph input
        # X = tf.placeholder(tf.float32, [None, num_input])
        # Y = tf.placeholder(tf.float32, [None, num_classes])
        # keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

    def load_data_set(self):
        ## Image preprocessing - Adding noise, reshaping, padding (maybe? prob not) etc.
        # Y = rel. position, pitch, yaw, roll
        batch_size = 32
        data_dir = ""       # Set directory
        train_ds = tf = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir, 
            validation_split = 0.2, 
            subset= "training",
            seed=123,
            image_size = (self.img_height, self.img_width),
            batch_size = batch_size)
        val_ds = tf = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir, 
            validation_split = 0.2, 
            subset= "training",
            seed=123,
            image_size = (self.img_height, self.img_width),
            batch_size = self.batch_size)

        ## Need to Know form of encoded labels for train_labels and test_labels
        return train_ds, val_ds, train_labels, test_labels
            
    def def_train_model(self, save_model = True):

        # Load dataset
        train_ds, val_ds, train_labels, test_labels = load_data_set()

        model = tf.keras.Sequential([
            layers.experimental.preprocessing.Rescaling(1./255),
            layers.Conv2D(32, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(64,activation = 'relu'),

            layers.Dense(self.num_classes)
        ])

        model.compile(
            optimizer='adam',
            loss=tf.losses.MeanSquaredError())

        history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels)

        ## Validate performance
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')

        test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

        # Save model 
        if save_model:
            model.save('/PackageRelModel')      # Alternatively can save as an H5 for computational savings. Test with wholesale first

    @staticmethod
    def predict(img):
        """
        Args:
        img = input png image from drone

        Output:
        rel_pos = relative position
        pitch = relative pitch
        yaw = relative yaw
        roll = relative roll
        """
        model =keras.models.load_model('/PackageRelModel')
        rel_pos, pitch, yaw, roll = model.predict(img)



import numpy as np
import tensorflow as tf

def load_data(images_path, labels_path, num_classes):
    # Load in data from .npz file.
    images = np.load(images_path)['arr_0']
    labels = np.load(labels_path)['arr_0']

    # Regularize images.
    images = np.array(images, dtype=np.float32) / 255
    labels = np.array(labels, dtype=np.uint8)

    # Reshape arrays.
    images = np.reshape(images, (-1, 28, 28, 1))
    labels = tf.one_hot(labels, num_classes)
    return (images, labels)
    
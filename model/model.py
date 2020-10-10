from preprocess import load_data
import tensorflow as tf
import numpy as np

class Model(tf.keras.Model):
    def __init__(self, learning_rate, batch_size, num_classes):
        super(Model, self).__init__()

        # Hyperparameters.
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # Trainable parameters.
        # Convolutional layer parameters.
        self.filter1 = tf.Variable(tf.random.truncated_normal([5, 5, 1, 16], stddev=0.1))
        self.filter1bias = tf.Variable(tf.random.truncated_normal([16], stddev=0.1))
        self.filter2 = tf.Variable(tf.random.truncated_normal([5, 5, 16, 20], stddev=0.1))
        self.filter2bias = tf.Variable(tf.random.truncated_normal([20], stddev=0.1))
        self.filter3 = tf.Variable(tf.random.truncated_normal([3, 3, 20, 20], stddev=0.1))
        self.filter3bias = tf.Variable(tf.random.truncated_normal([20], stddev=0.1))

        # Dense layer parameters.
        self.W1 = tf.Variable(tf.random.truncated_normal([4*20, 98], stddev=0.1))
        self.b1 = tf.Variable(tf.random.truncated_normal([1, 98], stddev=0.1))
        self.W2 = tf.Variable(tf.random.truncated_normal([98, 98], stddev=0.1))
        self.b2 = tf.Variable(tf.random.truncated_normal([1, 98], stddev=0.1))
        self.W3 = tf.Variable(tf.random.truncated_normal([98, 49], stddev=0.1))
        self.b3 = tf.Variable(tf.random.truncated_normal([1, 49], stddev=0.1))
        

    def call(self, inputs):
        # Block 1 - Convolution
        block1_conv = tf.nn.conv2d(inputs, self.filter1, strides=[1,2,2,1], padding="SAME")
        block1_conv_bias = tf.nn.bias_add(block1_conv, self.filter1bias)
        block1_mean, block1_variance = tf.nn.moments(block1_conv_bias, axes=[0,1,2])
        block1_norm = tf.nn.batch_normalization(block1_conv_bias, block1_mean, block1_variance, None, None, 1e-5)
        block1_relu = tf.nn.relu(block1_norm)
        block1_output = tf.nn.max_pool2d(block1_relu, [1,3,3,1], [1,2,2,1], padding="SAME")

        # Block 2 - Convolution
        block2_conv = tf.nn.conv2d(block1_output, self.filter2, strides=[1,2,2,1], padding="SAME")
        block2_conv_bias = tf.nn.bias_add(block2_conv, self.filter2bias)
        block2_mean, block2_variance = tf.nn.moments(block2_conv_bias, axes=[0,1,2])
        block2_norm = tf.nn.batch_normalization(block2_conv_bias, block2_mean, block2_variance, None, None, 1e-5)
        block2_relu = tf.nn.relu(block2_norm)
        block2_output = tf.nn.max_pool2d(block2_relu, [1,2,2,1], [1,2,2,1], padding="SAME")

        # Block 3 - Convolution
        block3_conv = tf.nn.conv2d(block2_output, self.filter3, strides=[1,1,1,1], padding="SAME")
        block3_conv_bias = tf.nn.bias_add(block3_conv, self.filter3bias)
        block3_mean, block3_variance = tf.nn.moments(block3_conv_bias, axes=[0,1,2])
        block3_norm = tf.nn.batch_normalization(block3_conv_bias, block3_mean, block3_variance, None, None, 1e-5)
        block3_relu = tf.nn.relu(block3_norm)
        block3_output = tf.reshape(block3_relu, [block3_relu.shape[0], -1])

        # Dense Layers
        dense1_output = tf.nn.dropout(tf.nn.relu(tf.matmul(block3_output, self.W1) + self.b1), rate=0.3)
        dense2_output = tf.nn.dropout(tf.nn.relu(tf.matmul(dense1_output, self.W2) + self.b2), rate=0.3)
        logits = tf.matmul(dense2_output, self.W3) + self.b3

        # Return.
        return logits


    def calc_loss(self, logits, labels):
        l = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        return l


    def accuracy(self, logits, labels):
        # NOTE: Taken from KMNIST.
        predictions = np.argmax(tf.nn.softmax(logits), axis=1)
        y = np.array([np.where(r==1)[0][0] for r in labels])
        accs = []
        for cls in range(49):
            mask = (y == cls)
            cls_acc = np.mean((predictions == cls)[mask])
            accs.append(cls_acc)
        accs = np.mean(accs)
        return accs


def train(model, train_inputs, train_labels):
    # Shuffle inputs.
    num_examples = train_inputs.shape[0]
    shuffle_idx = tf.random.shuffle(np.arange(num_examples))
    train_inputs, train_labels = tf.gather(train_inputs, shuffle_idx), tf.gather(train_labels, shuffle_idx)

    # Batched training.
    losses = []
    for i in range(num_examples // model.batch_size):
        if i % 100 == 0:
            print("Batch {}.".format(i))
        # Get batch start and end.
        start, end = i * model.batch_size, (i+1) * model.batch_size
        curr_inputs, curr_labels = train_inputs[start:end, :], train_labels[start:end]

        # Gradients.
        with tf.GradientTape() as tape:
            logits = model.call(curr_inputs)
            loss = model.calc_loss(logits, curr_labels)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        losses.append(loss)

    # Return.
    return losses


def test(model, test_inputs, test_labels):
    logits = model.call(test_inputs)
    acc = model.accuracy(logits, test_labels)
    return acc


def main():
    # Constants.
    TRAIN_INPUTS_PATH = "./data/k49-train-imgs.npz"
    TRAIN_LABELS_PATH = "./data/k49-train-labels.npz"
    TEST_INPUTS_PATH = "./data/k49-test-imgs.npz"
    TEST_LABELS_PATH = "./data/k49-test-labels.npz"

    # Hyperparams.
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    NUM_CLASSES = 49

    # Initialize model, get data.
    model = Model(LEARNING_RATE, BATCH_SIZE, NUM_CLASSES)
    train_inputs, train_labels = load_data(TRAIN_INPUTS_PATH, TRAIN_LABELS_PATH, NUM_CLASSES)
    test_inputs, test_labels = load_data(TEST_INPUTS_PATH, TEST_LABELS_PATH, NUM_CLASSES)

    # Train for NUM_EPOCHS
    for i in range(NUM_EPOCHS):
        print('Training Epoch {}'.format(i))
        train(model, test_inputs, test_labels)
        print('Testing Epoch {}'.format(i))
        acc = test(model, test_inputs, test_labels)
        print('Epoch {} accuracy: {}'.format(i, acc))


if __name__ == "__main__":
    main()

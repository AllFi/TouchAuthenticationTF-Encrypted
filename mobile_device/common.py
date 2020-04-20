"""Provide classes to perform private training and private prediction with
logistic regression"""
import tensorflow as tf
import tf_encrypted as tfe


class LogisticRegression:
    """Contains methods to build and train logistic regression."""

    def __init__(self, num_features, random_state=None):
        self.w = tfe.define_private_variable(
            tf.random_uniform([num_features, 1], -0.01, 0.01, seed=random_state)
        )
        self.w_masked = tfe.mask(self.w)
        self.b = tfe.define_private_variable(tf.zeros([1]))
        self.b_masked = tfe.mask(self.b)

    @property
    def weights(self):
        return self.w, self.b

    def forward(self, x):
        with tf.name_scope("forward"):
            out = tfe.matmul(x, self.w_masked) + self.b_masked
            y = tfe.sigmoid(out)
            return y

    def backward(self, x, dy, learning_rate=0.02):
        batch_size = x.shape.as_list()[0]
        with tf.name_scope("backward"):
            dw = tfe.matmul(tfe.transpose(x), dy) / batch_size
            db = tfe.reduce_sum(dy, axis=0) / batch_size
            assign_ops = [
                tfe.assign(self.w, self.w - dw * learning_rate),
                tfe.assign(self.b, self.b - db * learning_rate),
            ]
            return assign_ops

    def loss_grad(self, y, y_hat):
        with tf.name_scope("loss-grad"):
            dy = y_hat - y
            return dy

    def fit_batch(self, x, y):
        with tf.name_scope("fit-batch"):
            y_hat = self.forward(x)
            dy = self.loss_grad(y, y_hat)
            fit_batch_op = self.backward(x, dy)
            return fit_batch_op

    def fit(self, sess, x, y, num_batches):
        fit_batch_op = self.fit_batch(x, y)
        for batch in range(num_batches):
            print("Batch {0: >4d}".format(batch))
            sess.run(fit_batch_op, tag="fit-batch")

    def evaluate(self, sess, x, y, data_owner):
        """Return the accuracy"""

        def print_accuracy(y_hat, y) -> tf.Operation:
            with tf.name_scope("print-accuracy"):
                correct_prediction = tf.equal(tf.round(y_hat), y)
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                print_op = tf.print("Accuracy on {}:".format(data_owner.player_name), accuracy)
                return print_op

        with tf.name_scope("evaluate"):
            y_hat = self.forward(x)
            print_accuracy_op = tfe.define_output(
                data_owner.player_name, [y_hat, y], print_accuracy
            )

        sess.run(print_accuracy_op, tag="evaluate")


class DataOwner:
    """Contains code meant to be executed by a data owner Player."""

    def __init__(self, player_name, train_set, test_set, batch_size):
        self.player_name = player_name
        self.train_set = train_set
        self.test_set = test_set
        self.batch_size = batch_size

    @tfe.local_computation
    def provide_training_data(self):
        target = self.train_set.pop("target")
        dataset = tf.data.Dataset.from_tensor_slices((self.train_set.values, target.values)) \
            .repeat() \
            .batch(self.batch_size)

        train_set_iterator = dataset.make_one_shot_iterator()
        x, y = train_set_iterator.get_next()
        x = tf.reshape(x, [self.batch_size, self.train_set.shape[1]])
        y = tf.reshape(y, [self.batch_size, 1])
        return x, y

    @tfe.local_computation
    def provide_test_data(self):
        target = self.test_set.pop("target")
        dataset = tf.data.Dataset.from_tensor_slices((self.test_set.values, target.values)) \
            .batch(self.test_set.shape[0])

        test_set_iterator = dataset.make_one_shot_iterator()
        x, y = test_set_iterator.get_next()
        x = tf.reshape(x, [self.test_set.shape[0], self.test_set.shape[1]])
        y = tf.reshape(y, [self.test_set.shape[0], 1])
        return x, y


class ModelOwner:
    """Contains code meant to be executed by a model owner Player."""

    def __init__(self, player_name, features):
        self.player_name = player_name
        self.features = features

    @tfe.local_computation
    def receive_weights(self, *weights):
        features_with_weights = {self.features[i]: weights[0][0][i][0] for i in range(len(self.features))}
        return tf.print(features_with_weights)
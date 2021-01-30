import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np

def plot_history(history):
    loss = history.history['loss']
    acc = history.history['accuracy']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_accuracy']

    plt.figure(figsize=(10,4))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['Training','Validation'])

    plt.figure(figsize=(10,4))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(acc)
    plt.plot(val_acc)
    plt.legend(['Training','Validation'])

    plt.show()

# version 1
class myDropout(keras.layers.Dropout):
    """Applies Dropout to the input.
    Dropout consists in randomly setting
    a fraction `rate` of input units to 0 at each update during training time,
    which helps prevent overfitting.
    # Arguments
        rate: float between 0 and 1. Fraction of the input units to drop.
        noise_shape: 1D integer tensor representing the shape of the
            binary dropout mask that will be multiplied with the input.
            For instance, if your inputs have shape
            `(batch_size, timesteps, features)` and
            you want the dropout mask to be the same for all timesteps,
            you can use `noise_shape=(batch_size, 1, features)`.
        seed: A Python integer to use as random seed.
    # References
        - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](
           http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)
    """
    def __init__(self, rate, training=True, noise_shape=None, seed=None, name=None, **kwargs):
        super(myDropout, self).__init__(rate, noise_shape=None, seed=None,name = name,**kwargs)
        self.training = training

    def get_config(self):
        config = super(myDropout, self).get_config()
        config.update({"rate": self.rate})
        return config

    def call(self, inputs, training=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(inputs)

            def dropped_inputs():
                return K.dropout(inputs, self.rate, noise_shape,
                                 seed=self.seed)
            if not training:
                return K.in_train_phase(dropped_inputs, inputs, training=self.training)
            return K.in_train_phase(dropped_inputs, inputs, training=training)
        return inputs

# version 2
# class myDropout(keras.layers.Dropout):
#     """Applies Dropout to the input.
#     Dropout consists in randomly setting
#     a fraction `rate` of input units to 0 at each update during training time,
#     which helps prevent overfitting.
#     # Arguments
#         rate: float between 0 and 1. Fraction of the input units to drop.
#         noise_shape: 1D integer tensor representing the shape of the
#             binary dropout mask that will be multiplied with the input.
#             For instance, if your inputs have shape
#             `(batch_size, timesteps, features)` and
#             you want the dropout mask to be the same for all timesteps,
#             you can use `noise_shape=(batch_size, 1, features)`.
#         seed: A Python integer to use as random seed.
#     # References
#         - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](
#            http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)
#     """
#     def __init__(self, rate, training=True, noise_shape=None, seed=None, name=None, **kwargs):
#         super(myDropout, self).__init__(rate, noise_shape=None, seed=None,name = name,**kwargs)
#         self.training = training
#
#     def get_config(self):
#         config = super(myDropout, self).get_config()
#         config.update({"rate": self.rate})
#         # return config
#         return {"output": config, "name": self.name}
#
#
#     def call(self, inputs, training=None):
#         if 0. < self.rate < 1.:
#             noise_shape = self._get_noise_shape(inputs)
#
#             def dropped_inputs():
#                 return K.dropout(inputs, self.rate, noise_shape,
#                                  seed=self.seed)
#             if not training:
#                 return K.in_train_phase(dropped_inputs, inputs, training=self.training)
#             return K.in_train_phase(dropped_inputs, inputs, training=training)
#         return inputs

def MC_eval(model, nTimes, dataset_test, steps):

    accuracies = np.zeros(shape=(nTimes, ))
    losses = np.zeros(shape=(nTimes, ))
    for i in range(nTimes):
        score = model.evaluate(dataset_test, steps=steps, verbose=0)
        losses[i] = score[0]
        accuracies[i] = score[1]

    return accuracies, losses

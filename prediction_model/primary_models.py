import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

from prediction_model.dataset_preprocessor import process_input

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def df_to_dataset(dataframe, batch_size=32, shuffle=True):
    dataframe = dataframe.copy()
    response = dataframe.pop('y')    
    dataset = tf.data.Dataset.from_tensor_slices((dict(dataframe), response.values))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(dataframe))
    if batch_size:
        dataset = dataset.batch(batch_size)

    return dataset

def pack_features_vector(features, labels):
    features = tf.stack(list(features.values()), axis=1)

    return features, labels

def pack_labels_vector(features, labels):
    labels = tf.reshape(labels, [32, 1])
    labels = tf.transpose(labels)

    return features, labels

class LogisticLayer(keras.layers.Layer):
    def __init__(self, input_shape, trained, name=None, model_type=None, num_outputs=1):
        super(LogisticLayer, self).__init__()
        if trained:
            path = os.path.join(BASE_DIR, 'data', 'trained_model', 'weights_{0}_{1}.npy'.format(name, model_type))
            self.w = tf.Variable(np.load(path))
        else:
            w_init = tf.initializers.GlorotUniform()     # NOTICE : weight matrix itself contains bias
            self.w = tf.Variable(
                initial_value=w_init(shape=(input_shape, num_outputs), dtype='float32'),
                trainable=True)

    @tf.function
    def call(self, features):
        return tf.matmul(features, self.w)

class LogisticModel(tf.keras.Model):
    def __init__(self, input_shape, trained=False, name=None, model_type=None):
        super(LogisticModel, self).__init__()
        self.logistic_layer = LogisticLayer(input_shape, trained=trained, name=name, model_type=model_type)    #input shape = 146, 1

    @tf.function
    def call(self, features, training=False):
        x = self.logistic_layer(features)
        return x

optimizer = tf.optimizers.Adam(learning_rate=0.1)

def hypothesis(logit):
    return tf.divide(1.0, 1.0 + tf.exp(-1.0*logit))

def cost(logits, labels):
    hypo = hypothesis(logits)
    return -tf.reduce_mean(tf.math.log(hypo)*labels + tf.math.log(1.0-hypo)*(1.0-labels))

def train(model, features, labels):
    with tf.GradientTape() as t:
        current_cost = cost(model(features), labels)
    dW = t.gradient(current_cost, [model.logistic_layer.w])
    optimizer.apply_gradients(zip(dW, [model.logistic_layer.w]))
    
    return current_cost

def batch_accuracy(hypos, labels, batch_size=32):
    accuracy = 0
    predictor = zip(hypos, labels)
    for hypo, label in predictor:
        prediction = 0
        if hypo >= 0/5: prediction = 1

        if prediction == label: 
            accuracy += 1
    return accuracy/batch_size


def primary_model_train(name, model_type):    
    name = name.replace(' ', '-')
    df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'dataset', 'dataset_{0}_{1}.csv'.format(name, model_type)), dtype='float32')
    #df = df.sample(frac=1).reset_index(drop=True)   # shuffle
    
    batch_size = 32

    _, cut = divmod(len(df), batch_size)
    
    df = df.loc[:len(df)-cut-1]
    ds = df_to_dataset(df)
    dataset = ds.map(pack_features_vector)
 
    model = LogisticModel(146)
    
    costs = list()
    for features, labels in dataset:
        current_cost = train(model, features, labels)
        costs.append(current_cost.numpy())
    
    weights = model.logistic_layer.w.numpy()    
    np.save(os.path.join(BASE_DIR, 'data', 'trained_model', 'weights_{0}_{1}.npy'.format(name, model_type)), weights)

def primary_model_predict(name, model_type, model_input):
    name = name.replace(' ', '-')
    model = LogisticModel(146, trained=True, name=name, model_type=model_type)
    logit = model([process_input(model_input)])
    hypos = hypothesis(logit)

    return hypos.numpy()[0][0]

if __name__ == "__main__":
    name = "hide on bush"
    primary_model_train(name, "enemy")
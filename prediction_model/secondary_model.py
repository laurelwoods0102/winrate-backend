import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def df_to_dataset(dataframe, batch_size=32):
    response = dataframe.pop('y')    
    dataset = tf.data.Dataset.from_tensor_slices((dict(dataframe), response.values))
    dataset = dataset.batch(batch_size)

    return dataset

def pack_features_vector(features, labels):
    features = tf.stack(list(features.values()), axis=1)

    labels = tf.reshape(labels, [32, 1])
    labels = tf.transpose(labels)

    return features, labels

class LogisticLayer(keras.layers.Layer):
    def __init__(self, input_shape, trained, name=None, num_outputs=1):
        super(LogisticLayer, self).__init__()
        if trained:
            path = os.path.join(BASE_DIR, 'data', 'trained_model', 'weights_{0}_secondary.npy'.format(name))
            self.w = tf.Variable(np.load(path))
        else:
            w_init = tf.initializers.GlorotUniform()     # NOTICE : weight matrix itself contains bias
            self.w = tf.Variable(
                initial_value=w_init(shape=(input_shape, num_outputs), dtype=tf.float32),
                trainable=True)

    @tf.function
    def call(self, features):
        return tf.matmul(features, self.w)

class LogisticModel(tf.keras.Model):
    def __init__(self, input_shape, trained=False, name=None):
        super(LogisticModel, self).__init__()
        self.logistic_layer = LogisticLayer(input_shape, trained=trained, name=name)    #input shape = 146, 4

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

def secondary_model_train(name):
    name = name.replace(' ', '-')
    secondary_path = os.path.join(BASE_DIR, 'data', 'dataset', 'secondary')
    df_team = pd.read_csv(os.path.join(secondary_path, "dataset_{0}_team.csv".format(name)), dtype='float32')
    df_enemy = pd.read_csv(os.path.join(secondary_path, "dataset_{0}_enemy.csv".format(name)), dtype='float32')
    df_average = pd.read_csv(os.path.join(secondary_path, "dataset_{0}_average.csv".format(name)), dtype='float32')
    df_response = pd.read_csv(os.path.join(secondary_path, "dataset_{0}_response.csv".format(name)), dtype='float32')

    bias = np.array([[float(1) for i in range(len(df_response))]], dtype='f4')
    bias = pd.DataFrame(bias.T, columns=["bias"])
    
    df = pd.concat([bias, df_average, df_team, df_enemy, df_response], axis=1)

    batch_size = 32

    _, cut = divmod(len(df), batch_size)
    
    df = df.loc[:len(df)-cut-1]
    ds = df_to_dataset(df)
    dataset = ds.map(pack_features_vector)

    model = LogisticModel(4)
    
    costs = list()
    for features, labels in dataset:
        current_cost = train(model, features, labels)
        costs.append(current_cost.numpy())

    weights = model.logistic_layer.w.numpy()    
    np.save(os.path.join(BASE_DIR, 'data', 'trained_model', 'weights_{0}_secondary.npy'.format(name)), weights)

def secondary_model_predict(name, input_team, input_enemy):
    name = name.replace(' ', '-')
    average_table = pd.read_csv(os.path.join(BASE_DIR, 'data', 'dataset', 'secondary', 'dataset_{}_average_table.csv'.format(name)), dtype='float32')

    prediction = list()
    model = LogisticModel(4, trained=True, name=name)

    for i, average in enumerate(average_table.values.T.tolist()[0]):
        if average == 0: continue
        result = dict()
        dataset = list()

        dataset.append(1.0)
        dataset.append(input_team)
        dataset.append(input_enemy)
        dataset.append(average)

        logit = model([dataset])
        hypos = hypothesis(logit)

        result["champion"] = i
        result["predict"] = hypos.numpy()[0][0]
        prediction.append(result)
    
    return prediction

if __name__ == "__main__":
    name = "laurelwoods"
    secondary_model_predict("laurelwoods", None, None)
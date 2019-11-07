import json
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def average_dataset(name):
    name = name.replace(' ', '-')
    m = open(os.path.join(BASE_DIR, 'data', 'dataset', 'dataset_{}_pick_history.json'.format(name)))
    my_picks = json.load(m)

    average_table = [0 for i in range(145)]
    temp_win = [0 for i in range(145)]
    temp_total = [0 for i in range(145)]

    for data in my_picks:
        temp_total[data[1]] += 1
        if data[0] == 1:
            temp_win[data[1]] += 1

    for i in range(145):
        if temp_total[i] != 0:
            average_table[i] = temp_win[i]/temp_total[i]

    np_average_table = np.array([average_table], dtype='f4')
    df_average_table = pd.DataFrame(np_average_table.T, columns=["average_table"])

    path = os.path.join(BASE_DIR, 'data', 'dataset', 'secondary', 'dataset_{}_average_table.csv'.format(name))
    df_average_table.to_csv(path, index=False)
    
    average = list()
    for pick in my_picks:
        average.append(average_table[pick[1]])

    np_average = np.array([average])
    df_average = pd.DataFrame(np_average.T, columns=["average"])

    path = os.path.join(BASE_DIR, 'data', 'dataset', 'secondary', 'dataset_{}_average.csv'.format(name))
    df_average.to_csv(path, index=False)
    

class LogisticLayer(keras.layers.Layer):
    def __init__(self, weights):
        super(LogisticLayer, self).__init__()
        self.w = tf.convert_to_tensor(weights)

    @tf.function
    def call(self, features):
        return tf.matmul(features, self.w)

class LogisticModel(tf.keras.Model):
    def __init__(self, weights):
        super(LogisticModel, self).__init__()
        self.logistic_layer = LogisticLayer(weights)    #input shape = 146

    @tf.function
    def call(self, features, training=False):
        x = self.logistic_layer(features)
        return x

def hypothesis(logit):
    return tf.divide(1.0, 1.0 + tf.exp(-1.0*logit))

def secondary_dataset_generate(name, model_type):
    name = name.replace(' ', '-')
    weights = np.load(os.path.join(BASE_DIR, 'data', 'trained_model', 'weights_{0}_{1}.npy'.format(name, model_type)))

    df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'dataset', 'dataset_{0}_{1}.csv'.format(name, model_type)), dtype='float32')
    response = df.pop('y')
    
    model = LogisticModel(weights)

    predictions = list()
    for data in df.values:
        data = np.reshape(data, (1, 146))        
        logit = model(tf.convert_to_tensor(data))
        predict = hypothesis(logit).numpy()[0][0]
        predictions.append(predict)
    
    data = np.array([predictions])
    dataset = pd.DataFrame(data.T, columns=["predict_{}".format(model_type)])
    
    dataset.to_csv(os.path.join(BASE_DIR, 'data', 'dataset', 'secondary', 'dataset_{0}_{1}.csv'.format(name, model_type)), index=False)
    response.to_csv(os.path.join(BASE_DIR, 'data', 'dataset', 'secondary', 'dataset_{0}_{1}_response.csv'.format(name, model_type)), index=False, header=['y'])


if __name__ == "__main__":
    name = "hide on bush"
    #processor(name, "enemy")
    average_dataset(name)
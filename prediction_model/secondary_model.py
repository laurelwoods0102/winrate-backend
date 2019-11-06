import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow import keras

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
    def __init__(self, input_shape, num_outputs=1):
        super(LogisticLayer, self).__init__()
        w_init = tf.initializers.GlorotUniform()     # NOTICE : weight matrix itself contains bias
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_shape, num_outputs), dtype=tf.float32),
            trainable=True)

    @tf.function
    def call(self, features):
        return tf.matmul(features, self.w)

class LogisticModel(tf.keras.Model):
    def __init__(self, input_shape):
        super(LogisticModel, self).__init__()
        self.logistic_layer = LogisticLayer(input_shape)    #input shape = 146, 4

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


def KFoldValidation(name):
    df_team = pd.read_csv("./dataset/secondary/dataset_{0}_team.csv".format(name), dtype='float32')
    df_enemy = pd.read_csv("./dataset/secondary/dataset_{0}_enemy.csv".format(name), dtype='float32')
    df_average = pd.read_csv("./dataset/secondary/dataset_{0}_average.csv".format(name), dtype='float32')
    df_response = pd.read_csv("./dataset/secondary/dataset_{0}_response.csv".format(name), dtype='float32')

    bias = np.array([[float(1) for i in range(len(df_response))]], dtype='f4')
    bias = pd.DataFrame(bias.T, columns=["bias"])
    
    df = pd.concat([bias, df_average, df_team, df_enemy, df_response], axis=1)

    batch_size = 32

    train_len, test_size = divmod(len(df), batch_size)
    if test_size == 0:
        train_len -= 1
        test_size = batch_size

    train_df = df.loc[:train_len*batch_size-1]
    test_df = df.loc[train_len*batch_size-1:]

    models = list()
    train_cost = list()
    val_accuracy = list()

    kf = KFold(n_splits=train_len)
    for train_index, val_index in kf.split(train_df):
        model = LogisticModel(4)

        train_ds = df_to_dataset(train_df.iloc[train_index])
        train_dataset = train_ds.map(pack_features_vector)
        for features, labels in train_dataset:
            current_cost = train(model, features, labels)
            train_cost.append(current_cost.numpy())

        val_ds = df_to_dataset(train_df.iloc[val_index])
        val_dataset = val_ds.map(pack_features_vector)
        for features, labels in val_dataset:
            logits = model(features)
            hypos = hypothesis(logits)
            
            accuracy = batch_accuracy(tf.reshape(hypos, [1, 32]).numpy()[0], labels.numpy()[0])
            val_accuracy.append(accuracy)
        
        models.append(model)

    tc = np.array([train_cost])
    tc_df = pd.DataFrame(tc.T, columns=["train_cost"])
    tc_df.to_csv("./model_results/{0}_secondary_train_cost.csv".format(name), index=False)

    va = np.array([val_accuracy])
    va_df = pd.DataFrame(va.T, columns=["val_accuracy"])
    va_df.to_csv("./model_results/{0}_secondary_validation_accuracy.csv".format(name), index=False)

def processor(name):
    df_team = pd.read_csv("./dataset/secondary/dataset_{0}_team.csv".format(name), dtype='float32')
    df_enemy = pd.read_csv("./dataset/secondary/dataset_{0}_enemy.csv".format(name), dtype='float32')
    df_average = pd.read_csv("./dataset/secondary/dataset_{0}_average.csv".format(name), dtype='float32')
    df_response = pd.read_csv("./dataset/secondary/dataset_{0}_response.csv".format(name), dtype='float32')

    bias = np.array([[float(1) for i in range(len(df_response))]], dtype='f4')
    bias = pd.DataFrame(bias.T, columns=["bias"])
    
    df = pd.concat([bias, df_average, df_team, df_enemy, df_response], axis=1)

    batch_size = 32

    train_len, test_size = divmod(len(df), batch_size)
    if test_size == 0:
        train_len -= 1
        test_size = batch_size

    train_df = df.loc[:train_len*batch_size-1]
    test_df = df.loc[train_len*batch_size-1:]

    train_ds = df_to_dataset(train_df)
    train_dataset = train_ds.map(pack_features_vector)

    test_response = test_df.pop('y')
    test_dataset = zip(test_df.values, test_response.values)

    model = LogisticModel(4)
    
    costs = list()
    for features, labels in train_dataset:
        current_cost = train(model, features, labels)
        costs.append(current_cost.numpy())

    test_hypos = list()
    test_labels = list()

    for features, label in test_dataset:
        features = tf.convert_to_tensor([features])
        logit = model(features)
        hypo = hypothesis(logit)

        test_hypos.append(hypo)
        test_labels.append(label) 
    
    test_accuracy = batch_accuracy(test_hypos, test_labels, batch_size=len(test_hypos))
    print("test accuracy : ", test_accuracy)

    test_df.to_csv("./model_results/{0}_secondary_test_dataset.csv".format(name), index=False)

    weights = model.logistic_layer.w.numpy()
    np.save('./trained_model/weights_secondary_{0}.npy'.format(name), weights)

if __name__ == "__main__":
    name = "hide on bush".replace(" ", "-")
    #KFoldValidation(name)
    processor(name)
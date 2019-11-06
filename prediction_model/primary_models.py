import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow import keras

def df_to_dataset(dataframe, batch_size=32):
    dataframe = dataframe.copy()
    response = dataframe.pop('y')    
    dataset = tf.data.Dataset.from_tensor_slices((dict(dataframe), response.values))

    #if shuffle:
        #dataset = dataset.shuffle(buffer_size=len(dataframe))
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
    def __init__(self, input_shape, num_outputs=1):
        super(LogisticLayer, self).__init__()
        w_init = tf.initializers.GlorotUniform()     # NOTICE : weight matrix itself contains bias
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_shape, num_outputs), dtype='float32'),
            trainable=True)

    @tf.function
    def call(self, features):
        return tf.matmul(features, self.w)

class LogisticModel(tf.keras.Model):
    def __init__(self, input_shape):
        super(LogisticModel, self).__init__()
        self.logistic_layer = LogisticLayer(input_shape)    #input shape = 146, 1

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


def KFoldValidation(name, model_type):
    df = pd.read_csv("./dataset/dataset_{0}_{1}.csv".format(name, model_type), dtype='float32')
    df = df.sample(frac=1).reset_index(drop=True)   # shuffle

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
        model = LogisticModel(146)

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
            
            accuracy = batch_accuracy(tf.reshape(hypos, [1, 32]).numpy()[0], labels.numpy())
            val_accuracy.append(accuracy)
        
        models.append(model)
    
    with open('./model_results/{0}_{1}_train_cost.txt'.format(name, model_type), 'w') as f:
        for tc in train_cost:
            f.write(str(tc))
            f.write('\n')

    with open('./model_results/{0}_{1}_validation_accuracy.txt'.format(name, model_type), 'w') as g:
        for va in val_accuracy:
            g.write(str(va))
            g.write('\n')

def processor(name, model_type):
    df = pd.read_csv("./dataset/dataset_{0}_{1}.csv".format(name, model_type))
    df = df.sample(frac=1).reset_index(drop=True)   # shuffle
    df = df.astype('float32')
    
    batch_size = 32

    train_len, test_size = divmod(len(df), batch_size)
    if test_size == 0:
        train_len -= 1
        test_size = batch_size

    train_df = df.loc[:train_len*batch_size-1]
    test_df = df.loc[train_len*batch_size-1:]

    train_ds = df_to_dataset(train_df)
    train_dataset = train_ds.map(pack_features_vector)
    #train_dataset = train_ds.map(pack_labels_vector)

    test_response = test_df.pop('y')
    test_dataset = zip(test_df.values, test_response.values)

 
    model = LogisticModel(146)
    
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

    test_df.to_csv("./model_results/{0}_{1}_test_dataset.csv".format(name, model_type), index=False)
    
    weights = model.logistic_layer.w.numpy()
    np.save('./trained_model/weights_{0}_{1}.npy'.format(name, model_type), weights)
    
if __name__ == "__main__":
    name = "hide on bush".replace(' ', '-')
    KFoldValidation(name, "enemy")
    processor(name, "enemy")
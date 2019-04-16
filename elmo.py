import numpy as np
import tensorflow as tf
from util import *
import random
import tensorflow_hub as hub
import pandas as pd

# Set file names
file_train_instances = "train_stances.csv"
file_train_bodies = "train_bodies.csv"
file_test_instances = "test_stances_unlabeled.csv"
file_test_bodies = "test_bodies.csv"
file_predictions = 'predictions_test.csv'
# Initialise hyperparameters
r = random.Random()
lim_unigram = 5000
target_size = 4
hidden_size = 100
train_keep_prob = 0.6
l2_alpha = 0.00001
learn_rate = 0.01
clip_ratio = 5
batch_size_train = 500
epochs = 90

# Load data sets
raw_train = FNCData(file_train_instances, file_train_bodies)
raw_test = FNCData(file_test_instances, file_test_bodies)
n_train = len(raw_train.instances)

elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

inst = pd.DataFrame(raw_train.instances)
unique_inst = inst.drop_duplicates(subset=['Body ID'])

# getting heads and bodies
heads = list(unique_inst.Headline.values)
bodies = [raw_train.bodies[i] for i in unique_inst['Body ID'].values]

head_emb = elmo(heads, signature="default", as_dict=True)
body_emb = elmo(bodies, signature="default", as_dict=True)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(head_emb['default'][0]))

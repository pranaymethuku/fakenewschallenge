import numpy as np
from util import *
import random
#import tensorflow as tf
#import tensorflow_hub as hub
import pandas as pd
from feature_helper import word_overlap_features
from feature_helper import refuting_features, polarity_features, hand_features, gen_or_load_feats

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

#elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

inst = pd.DataFrame(raw_train.instances)
unique_inst = inst.drop_duplicates(subset=['Body ID'])

# getting heads and bodies
heads = list(unique_inst.Headline.values)
bodies = [raw_train.bodies[i] for i in unique_inst['Body ID'].values]
stances = list(unique_inst.Stance.values)


##these are derived from baseline
#These are 44 features
X_overlap = gen_or_load_feats(word_overlap_features, heads, bodies, "features/overlap.training.npy")
X_refuting = gen_or_load_feats(refuting_features, heads, bodies, "features/refuting.training.npy")
X_polarity = gen_or_load_feats(polarity_features, heads, bodies, "features/polarity.training.npy")
X_hand = gen_or_load_feats(hand_features, heads, bodies, "features/hand.training.npy")
##TODO :: please fix this appropriately.
'''
concatenate elmo of headline, body and and crafted features
'''
X = np.c_[X_hand, X_polarity, X_refuting, X_overlap]  
head_emb = elmo(heads, signature="default", as_dict=True)
body_emb = elmo(bodies, signature="default", as_dict=True)


# Define model - we are only giving the trianing option

# Create placeholders
##TODO :: set the feature_size appropriately
features_pl = tf.placeholder(tf.float32, [None, feature_size], 'features')
stances_pl = tf.placeholder(tf.int64, [None], 'stances')
keep_prob_pl = tf.placeholder(tf.float32)

# Infer batch size
batch_size = tf.shape(features_pl)[0]

# Define multi-layer perceptron
hidden_layer = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.linear(features_pl, hidden_size)), keep_prob=keep_prob_pl)
logits_flat = tf.nn.dropout(tf.contrib.layers.linear(hidden_layer, target_size), keep_prob=keep_prob_pl)
logits = tf.reshape(logits_flat, [batch_size, target_size])

# Define L2 loss
tf_vars = tf.trainable_variables()
l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf_vars if 'bias' not in v.name]) * l2_alpha

# Define overall loss
loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, stances_pl) + l2_loss)

# Define prediction
softmaxed_logits = tf.nn.softmax(logits)
predict = tf.arg_max(softmaxed_logits, 1)


# Define optimiser
opt_func = tf.train.AdamOptimizer(learn_rate)
grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tf_vars), clip_ratio)
opt_op = opt_func.apply_gradients(zip(grads, tf_vars))

# Perform training
##TO DO :: Fix train/test pointers!
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        total_loss = 0
        indices = list(range(n_train))
        r.shuffle(indices)

        for i in range(n_train // batch_size_train):
            batch_indices = indices[i * batch_size_train: (i + 1) * batch_size_train]
            batch_features = [train_set[i] for i in batch_indices]
            batch_stances = [train_stances[i] for i in batch_indices]

            batch_feed_dict = {features_pl: batch_features, stances_pl: batch_stances, keep_prob_pl: train_keep_prob}
            _, current_loss = sess.run([opt_op, loss], feed_dict=batch_feed_dict)
            total_loss += current_loss


    # Predict
    test_feed_dict = {features_pl: test_set, keep_prob_pl: 1.0}
    test_pred = sess.run(predict, feed_dict=test_feed_dict)


# Save predictions
save_predictions(test_pred, file_predictions)

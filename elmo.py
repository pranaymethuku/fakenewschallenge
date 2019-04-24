import numpy as np
from util import FNCData, save_predictions
import random
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from feature_helper import word_overlap_features
from feature_helper import refuting_features, polarity_features, hand_features, gen_feats

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

def get_train_dataset(train_df, raw_train):
    # getting heads and bodies
    train_heads = list(train_df.Headline.values)
    train_bodies = [raw_train.bodies[i] for i in train_df['Body ID'].values]
    train_stances = list(train_df.Stance.values)

    ##these are derived from baseline
    #These are 44 features
    train_feat_overlap = gen_feats(word_overlap_features, train_heads, train_bodies)
    train_feat_refuting = gen_feats(refuting_features, train_heads, train_bodies)
    train_feat_polarity = gen_feats(polarity_features, train_heads, train_bodies)
    train_feat_hand = gen_feats(hand_features, train_heads, train_bodies)
    
    # concatenate elmo of headline, body and and crafted features
    X = np.c_[train_feat_hand, train_feat_polarity, train_feat_refuting, train_feat_overlap]
    train_feat_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    head_emb = elmo(train_heads, signature="default", as_dict=True)["default"]
    body_emb = elmo(train_bodies, signature="default", as_dict=True)["default"]
    train_set = tf.concat([train_feat_tensor, head_emb, body_emb], axis=1)
    return train_set, train_stances

def get_test_dataset(test_df, raw_test):
    # getting heads and bodies
    test_heads = list(test_df.Headline.values)
    test_bodies = [raw_test.bodies[i] for i in test_df['Body ID'].values]

    ##these are derived from baseline
    #These are 44 features
    test_feat_overlap = gen_feats(word_overlap_features, test_heads, test_bodies)
    test_feat_refuting = gen_feats(refuting_features, test_heads, test_bodies)
    test_feat_polarity = gen_feats(polarity_features, test_heads, test_bodies)
    test_feat_hand = gen_feats(hand_features, test_heads, test_bodies)
    
    # concatenate elmo of headline, body and and crafted features
    X = np.c_[test_feat_hand, test_feat_polarity, test_feat_refuting, test_feat_overlap]
    test_feat_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    head_emb = elmo(test_heads, signature="default", as_dict=True)["default"]
    body_emb = elmo(test_bodies, signature="default", as_dict=True)["default"]
    test_set = tf.concat([test_feat_tensor, head_emb, body_emb], axis=1)
    return test_set

# set up train dataset
train_inst = pd.DataFrame(raw_train.instances)
train_df = train_inst.drop_duplicates(subset=['Body ID'])
train_set, train_stances = get_train_dataset(train_df, raw_train)
feature_size = train_set.shape[1]

# set up test dataset
test_inst = pd.DataFrame(raw_test.instances)
test_df = test_inst.drop_duplicates(subset=['Body ID'])
test_set = get_test_dataset(test_df, raw_test)

# Define model - we are only giving the training option

# Create placeholders
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

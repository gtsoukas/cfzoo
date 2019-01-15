# TODOs:
#   * add tuning grid for model hyper parameters
#   * maybe write training time to file
#   * stick with one logging mechanism

import argparse
import datetime
import logging
import numpy as np
import time
import tensorflow as tf
from tensorflow.contrib.factorization.python.ops import factorization_ops
from sklearn.datasets import load_svmlight_file

DATA_TYPE = np.float32
RANK = 32
N_ITER = 3

LOG_RATINGS = 0
LINEAR_RATINGS = 1
LINEAR_OBS_W = 100.0

parser = argparse.ArgumentParser(
    description='Fit and predict with Ben Fredericksons implicit.')
parser.add_argument('train', help='Training data file')
parser.add_argument('test', help='Testing data file')
parser.add_argument('negatives', help='Negatives data file')
parser.add_argument('ranking'
    , help='Output file with 10 ranked samples which contains 9 or 10 negatives'
        + ' and potentially one positive sample.')
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG
    , format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# https://github.com/GoogleCloudPlatform/tensorflow-recommendation-wals/blob/d42e887a3f0c36d0653ca32ebefe79658cb38767/wals_ml_engine/trainer/wals.py#L110

def simple_train(model, input_tensor, num_iterations):
    """Helper function to train model on input for num_iterations.
    Args:
    model:            WALSModel instance
    input_tensor:     SparseTensor for input ratings matrix
    num_iterations:   number of row/column updates to run
    Returns:
    tensorflow session, for evaluating results
    """
    sess = tf.Session(graph=input_tensor.graph)
    with input_tensor.graph.as_default():
        row_update_op = model.update_row_factors(sp_input=input_tensor)[1]
        col_update_op = model.update_col_factors(sp_input=input_tensor)[1]
        sess.run(model.initialize_op)
        sess.run(model.worker_init)
        for _ in range(num_iterations):
            sess.run(model.row_update_prep_gramian_op)
            sess.run(model.initialize_row_update_op)
            sess.run(row_update_op)
            sess.run(model.col_update_prep_gramian_op)
            sess.run(model.initialize_col_update_op)
            sess.run(col_update_op)
    return sess

def wals_model(data, dim, reg, unobs, weights=False,
               wt_type=LINEAR_RATINGS, feature_wt_exp=None,
               obs_wt=LINEAR_OBS_W):
    """Create the WALSModel and input, row and col factor tensors.
    Args:
    data:           scipy coo_matrix of item ratings
    dim:            number of latent factors
    reg:            regularization constant
    unobs:          unobserved item weight
    weights:        True: set obs weights, False: obs weights = unobs weights
    wt_type:        feature weight type: linear (0) or log (1)
    feature_wt_exp: feature weight exponent constant
    obs_wt:         feature weight linear factor constant
    Returns:
    input_tensor:   tensor holding the input ratings matrix
    row_factor:     tensor for row_factor
    col_factor:     tensor for col_factor
    model:          WALSModel instance
    """
    row_wts = None
    col_wts = None
    num_rows = data.shape[0]
    num_cols = data.shape[1]
    if weights:
        assert feature_wt_exp is not None
        row_wts = np.ones(num_rows)
        col_wts = make_wts(data, wt_type, obs_wt, feature_wt_exp, 0)
    row_factor = None
    col_factor = None
    with tf.Graph().as_default():
        input_tensor = tf.SparseTensor(indices=np.mat([data.row,
                                            data.col]).transpose(),
                                       values=data.data,
                                       dense_shape=data.shape)
        model = factorization_ops.WALSModel(num_rows, num_cols, dim,
                                            unobserved_weight=unobs,
                                            regularization=reg,
                                            row_weights=row_wts,
                                            col_weights=col_wts)
        # retrieve the row and column factors
        row_factor = model.row_factors[0]
        col_factor = model.col_factors[0]
    return input_tensor, row_factor, col_factor, model

logging.debug("Loading data")
start = time.time()
train, _ = load_svmlight_file(args.train, dtype=DATA_TYPE)
n_users, n_items = train.shape
test, _ = load_svmlight_file(args.test, n_features=n_items, dtype=DATA_TYPE)
negatives, _ = load_svmlight_file(args.negatives, n_features=n_items
    , dtype=DATA_TYPE)
logging.debug("Loading data took %0.2fs", time.time() - start)

logging.debug("Learning model")
start = time.time()
# generate model
input_tensor, row_factor, col_factor, model = wals_model(data=train.tocoo(),
                                                            dim=RANK,
                                                            reg=0.0,
                                                            unobs=1.0)
# factorize matrix
session = simple_train(model, input_tensor, num_iterations=N_ITER)
logging.debug("Learning model took %0.2fs", time.time() - start)

logging.debug("Writing rankings to file " + args.ranking)
start = time.time()
users = session.run(row_factor)
items = session.run(col_factor)
test2 = test + negatives
with open(args.ranking, 'w') as f:
    for user in range(n_users):
        items_to_rank = list(test2[user].nonzero()[1])
        ranking = []
        for item in items_to_rank:
            ranking.append((item, np.dot(users[user], items[item])))
        ranking10 = sorted(ranking, key=lambda tup: tup[1], reverse=True)[:10]
        f.write(str(user) + ', ' + str(ranking10) + '\n')

logging.debug("Writing rankings took %0.2fs", time.time() - start)

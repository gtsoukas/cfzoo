import argparse
import logging
import numpy as np
import time

import implicit
from implicit.nearest_neighbours import bm25_weight
import scipy.stats as stats
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import ParameterSampler

DATA_TYPE = np.float32
RANK = 32
N_ITER = 3

parser = argparse.ArgumentParser(
    description='Fit and predict with Ben Fredericksons implicit.')
parser.add_argument('train', help='training data file')
parser.add_argument('test', help='testing data file')
parser.add_argument('negatives', help='negatives data file')
parser.add_argument('ranking'
    , help='output file with 10 ranked samples which contains 9 or 10 negatives'
        + ' and potentially one positive sample.')
parser.add_argument('--validation-set'
    , dest='validation_set'
    , help='validaton data file'
    , default=None)
args = parser.parse_args()

logging.basicConfig(level=logging.INFO
    , format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logging.debug("Loading data")
start = time.time()
train, _ = load_svmlight_file(args.train, dtype=DATA_TYPE)
n_users, n_items = train.shape
test, _ = load_svmlight_file(args.test, n_features=n_items, dtype=DATA_TYPE)
if args.validation_set is not None:
    valid, _ = load_svmlight_file(args.validation_set, n_features=n_items
        , dtype=DATA_TYPE)
negatives, _ = load_svmlight_file(args.negatives, n_features=n_items
    , dtype=DATA_TYPE)
logging.debug("Loading data took %0.2fs", time.time() - start)

# TODO
def evaluate_model(model, positives, negatives):
    "Returns HR@10"
    valid2 = valid + negatives
    n_users = positives.shape[0]
    scores = []
    for user in range(n_users):
        items_to_rank = list(valid2[user].nonzero()[1])
        ranking = model.rank_items(userid=user, user_items=train
            , selected_items=items_to_rank)
        ranking10 = sorted(ranking, key=lambda tup: tup[1], reverse=True)[:10]
        tp_item = valid[user].nonzero()[1][0]
        if tp_item in [x[0] for x in ranking10]:
            scores.append(1.0)
        else:
            scores.append(0.0)
    return np.mean(scores)

rank = RANK
reg = 0.0
n_iter = N_ITER
bm25_K1 = 100
bm25_B = .8

if args.validation_set is not None:
    logging.info("Tuning hyperparamteters")

    # TODO: Separately list "budget" and regulariztion parameters
    param_dist = {
        'reg': stats.loguniform(0.0001, 10),
        'num_factors': list(range(2, 33)),
        'n_iter':list(range(3, 40)),
        'bm25_K1': stats.uniform(0.001, 200),
        'bm25_B': stats.uniform(0.001, 1)
    }
    param_list = list(ParameterSampler(param_dist, n_iter=100
                                   #,random_state=rng
                                  ))
    best_score = None
    best_params = None
    for i, params in enumerate(param_list):
        model = implicit.als.AlternatingLeastSquares(
            factors=params['num_factors'],
            regularization=params['reg'],
            use_cg=False,
            iterations=params['n_iter'],
            calculate_training_loss=True,
            num_threads=8
            )
        items_users = bm25_weight(train, params['bm25_K1']
            , params['bm25_B']).T.tocsr()
        model.fit(items_users, show_progress = False)
        score = evaluate_model(model, valid, negatives)
        # TODO: investigate how to handle uncertainty, i.e. when is a model
        # truly better and when is it just randomly better.
        new_best = ''
        if best_score is None or score > best_score:
            best_score = score
            best_params = params
            new_best = '*'

        logging.info(f'{i}, {params}, {score}{new_best}')

    logging.info('Best score: {:4.2f}%'.format(100*best_score))
    logging.info(f'Best params: {params}')
    # prepare data set for training final model on full data
    train = train + valid

    rank = best_params['num_factors']
    reg = best_params['reg']
    n_iter = best_params['n_iter']
    bm25_K1 = best_params['bm25_K1']
    bm25_B = best_params['bm25_B']



model = implicit.als.AlternatingLeastSquares(
  factors=rank,
  regularization=reg,
  use_cg=False,
  iterations=n_iter,
  calculate_training_loss=True,
  num_threads=8
  )

logging.debug("Weighting matrix by bm25_weight")
items_users = bm25_weight(train, bm25_K1, bm25_B).T.tocsr()
logging.debug("Learning model")
start = time.time()
model.fit(items_users, show_progress = False)
logging.debug("Learning model took %0.2fs", time.time() - start)

logging.debug("Writing rankings to file %s", args.ranking)
start = time.time()
test2 = test + negatives
with open(args.ranking, 'w') as f:
    for user in range(n_users):
        items_to_rank = list(test2[user].nonzero()[1])
        ranking = model.rank_items(userid=user, user_items=train
            , selected_items=items_to_rank)
        ranking10 = sorted(ranking, key=lambda tup: tup[1], reverse=True)[:10]
        f.write(str(user) + ', ' + str(ranking10) + '\n')

logging.debug("Writing rankings took %0.2fs", time.time() - start)

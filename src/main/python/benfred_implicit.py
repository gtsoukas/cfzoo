# TODOs:
#   * add tuning grid for model hyper parameters
#   * maybe write training time to file

import argparse
import implicit
import logging
import numpy as np
import time
from sklearn.datasets import load_svmlight_file

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
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG
    , format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logging.debug("Loading data")
start = time.time()
train, _ = load_svmlight_file(args.train, dtype=DATA_TYPE)
n_users, n_items = train.shape
test, _ = load_svmlight_file(args.test, n_features=n_items, dtype=DATA_TYPE)
negatives, _ = load_svmlight_file(args.negatives, n_features=n_items
    , dtype=DATA_TYPE)
logging.debug("Loading data took %0.2fs", time.time() - start)

model = implicit.als.AlternatingLeastSquares(
  factors=RANK,
  regularization=0.0,
  use_cg=False,
  iterations=N_ITER,
  calculate_training_loss=True,
  num_threads=8
  )

items_users = train.T.tocsr()
logging.debug("Learning model")
start = time.time()
model.fit(items_users)
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

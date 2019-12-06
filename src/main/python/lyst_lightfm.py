# TODOs:
#   * add tuning grid for model hyper parameters
#   * maybe write training time to file

import argparse
import logging
import numpy as np
import time

from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from sklearn.datasets import load_svmlight_file

DATA_TYPE = np.float32
RANK = 32
N_ITER = 10

parser = argparse.ArgumentParser(
    description='Fit and predict with LightFM.')
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

model = LightFM(
    no_components=RANK,
    learning_rate=0.05,
    loss='bpr'
    )

logging.debug("Learning model")
start = time.time()
model.fit(train, epochs=N_ITER)
logging.debug("Learning model took %0.2fs", time.time() - start)

test_precision = precision_at_k(model, test, k=5).mean()
logging.info("Test precision@5: {:4.2f}%".format(100*test_precision))

logging.debug("Writing rankings to file %s", args.ranking)
start = time.time()
test2 = test + negatives
with open(args.ranking, 'w') as f:
    for user in range(n_users):
        items_to_rank = list(test2[user].nonzero()[1])
        scores = model.predict(user, items_to_rank)
        ranking = list(zip(items_to_rank,scores))
        ranking10 = sorted(ranking, key=lambda tup: tup[1], reverse=True)[:10]
        f.write(str(user) + ', ' + str(ranking10) + '\n')

logging.debug("Writing rankings took %0.2fs", time.time() - start)

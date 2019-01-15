# TODOs:
#   * Write evaluation report to file.

import argparse
import logging
import numpy as np
import time
from sklearn.datasets import load_svmlight_file

DATA_TYPE=np.float32

def parse_list_of_tuples(s):
    tuples = s.strip('[]').split('), ')
    out = []
    for x in tuples:
        a, b = x.strip('()').split(',')
        out.append((int(a), np.float32(b)))
    return out

parser = argparse.ArgumentParser(
    description='Goodnes of fit evaluation of WRMF generated rankings')
parser.add_argument('train'
    , help='Training data file. Is inly used for counting the number of items.')
parser.add_argument('test', help='Testing data file')
parser.add_argument('ranking'
    , help='File with 10 ranked samples which contains 9 or 10 negatives and'
        + ' potentially one positive sample.')
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG
    , format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logging.debug("Loading data")
start = time.time()
# TODO, train is only loaded to get the number of items
train, _ = load_svmlight_file(args.train, dtype=DATA_TYPE)
n_users, n_items = train.shape
test, _ = load_svmlight_file(args.test, n_features=n_items, dtype=DATA_TYPE)

def scan_ranking_file():
    with open(args.ranking, 'r') as f:
        for l in f:
            x = l.strip('\n').split(', ', 1)
            user = int(x[0])
            ranking = parse_list_of_tuples(x[1])
            yield user, ranking

rankings = dict()
for u, r in scan_ranking_file():
    assert u not in rankings
    rankings[u] = r
logging.debug("Loading data took %0.2fs", time.time() - start)

logging.debug("Evaluating rankings")
start = time.time()
scores = np.zeros(n_users, dtype=np.float64)
for user in range(n_users):
    if user in rankings:
        r = list(map(lambda x: x[0], rankings[user]))
        p = list(test[user].nonzero()[1])[0]
        scores[user] = 1.0 if p in r else 0.0
    else:
        logging.warn(
            "missing user %d in ranking file %s", user, args.ranking)

logging.info("HR@10: {:4.2f}%".format(100*scores.mean()))
logging.debug("Evaluating rankings took %0.2fs", time.time() - start)

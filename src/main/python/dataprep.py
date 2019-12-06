# TODOs:
#   * Maybe write ID-index mapping to files.
#   * Speed up negative sampling.
#   * How to handle items that do not get sampled in the test data?
#     This may affect a large fraction of the items.

import argparse
import logging
import pandas as pd
import numpy as np
import scipy.sparse as sparse
import time
from sklearn import datasets
from sklearn.datasets import dump_svmlight_file

TRAIN_FILE = 'train.svm'
TEST_FILE = 'test.svm'
NEGATIVES_FILE = 'negatives.svm'
NUM_NEGATIVES = 99

parser = argparse.ArgumentParser(description='cfzoo data preparation')
parser.add_argument('inputfile',
    help='One line per interaction where the first column is a userid and the '
        + 'second column is an itemid. Each user-item combination must apear'
        + ' only once.')
parser.add_argument('--separator', default='\t')
parser.add_argument('--encoding', default='UTF-8', help='Inputfile encoding.')
parser.add_argument('--skip-headers', dest='headers', action='store_true',
    help='If flag is present, the first line of the inputfile is skipped.')
parser.set_defaults(headers=False)
#parser.add_argument('--max_sparsity', default=0.99)
#parser.add_argument('--seed', default=12345)
args = parser.parse_args()


logging.basicConfig(level=logging.DEBUG
    , format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# inspiration from: https://stackoverflow.com/questions/50665681/finding-n-random-zero-elements-from-a-scipy-sparse-matrix
# TODO: parallelize, for -> map
def sample_negatives(user_item_csr_matrix, n):
    """
    Randomly sample n zero items per user.
    Returns a list of item ids per user.
    Assumes highly sparse matrix.
    """
    num_users = user_item_csr_matrix.shape[0]
    num_items = user_item_csr_matrix.shape[1]
    negatives = sparse.lil_matrix(user_item_csr_matrix.shape
        , dtype=user_item_csr_matrix.dtype)
    nonzero_or_sampled = set(zip(*user_item_csr_matrix.nonzero()))
    def sample_zero_forever(user):
        while True:
            item = np.random.randint(0, num_items)
            if (user, item) not in nonzero_or_sampled:
                yield (user, item)
                nonzero_or_sampled.add((user, item))
    for user in range(num_users):
        itr = sample_zero_forever(user)
        for j in [next(itr) for _ in range(n)]:
            negatives[j] = 1.0
    return negatives.tocsr()

#TODO:
#   * Handle explicit 0s
#   * Maybe throw exception when no item is found for a user
#   * parallelize, for -> map
def one_split(user_item_csr_matrix):
    """
    Hold out exactly one, random non-zero item per user.
    Assumption: each user has at least one item interaction.
    """
    train = user_item_csr_matrix.copy().tolil()
    test = sparse.csr_matrix(user_item_csr_matrix.shape
        , dtype=user_item_csr_matrix.dtype).tolil()
    num_users = user_item_csr_matrix.shape[0]
    num_items = user_item_csr_matrix.shape[1]
    for user in range(num_users):
        # sample one non-zero item for user u
        nonzero_items = user_item_csr_matrix[user,:].nonzero()[1]
        random_nonzero_item = np.random.choice(nonzero_items)
        train[user, random_nonzero_item] = 0.0
        test[user, random_nonzero_item] = user_item_csr_matrix[user
            , random_nonzero_item]
    # Test and training are disjoint
    # assert(train.multiply(test).nnz == 0)
    return train.tocsr(), test.tocsr()


def scan_file():
    with open(args.inputfile, encoding=args.encoding) as in_file:
        if args.headers:
            next(in_file)
        for l in in_file:
            yield l.strip().split(args.separator)

logging.debug("Reading input data")
start = time.time()
input = [(x[0], x[1]) for x in scan_file()]
logging.debug("Reading input data took %0.2fs", time.time() - start)

df = pd.DataFrame(input, columns=['user_id', 'item_id'])

n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]

logging.info('Number of users: %d', n_users)
logging.info('Number of items: %d', n_items)
logging.info('Sparsity: %f', 1.0 - float(df.shape[0]) / float(n_users*n_items))

# TODO: maybe reduce sparsity to predefined threshold?

# Create mappings
logging.debug("Mapping IDs to indices")
start = time.time()
item_id_to_idx = {}
idx_to_item_id = {}
for (idx, item_id) in enumerate(df.item_id.unique().tolist()):
    item_id_to_idx[item_id] = idx
    idx_to_item_id[idx] = item_id

user_id_to_idx = {}
idx_to_user_id = {}
for (idx, user_id) in enumerate(df.user_id.unique().tolist()):
    user_id_to_idx[user_id] = idx
    idx_to_user_id[idx] = user_id

logging.debug("Mapping IDs to indices took %0.2fs", time.time() - start)

def map_ids(row, mapper):
    return mapper[row]

logging.debug("Converting data to csr matrix")
start = time.time()
# TODO: Try to reduce memory consumption here
I = df.user_id.apply(map_ids, args=[user_id_to_idx]).values
J = df.item_id.apply(map_ids, args=[item_id_to_idx]).values
V = np.ones(I.shape[0])
likes = sparse.coo_matrix((V, (I, J)), dtype=np.float32)
likes = likes.tocsr()
logging.debug("Converting data to csr matrix took %0.2fs", time.time() - start)

logging.debug("Sampling %s negatives", NUM_NEGATIVES)
start = time.time()
negatives = sample_negatives(likes, NUM_NEGATIVES)
logging.debug("Sampling negatives took %0.2fs", time.time() - start)

logging.debug("Splitting off test data")
start = time.time()
train, test = one_split(likes)
logging.debug("Splitting off test data took %0.2fs", time.time() - start)

test_item_scores = np.sum(test, 0).A1
num_items_test = len(test_item_scores.nonzero()[0])
logging.info("Share of items not in test set: %4.3f"
    , (n_items - num_items_test)/n_items)

# using user index as label in libsvm-file
logging.debug("Writing training data to file %s", TRAIN_FILE)
start = time.time()
dump_svmlight_file(X=train, y=list(range(n_users)), f=TRAIN_FILE)
logging.debug("Writing training data took %0.2fs", time.time() - start)

logging.debug("Writing testing data to file %s", TEST_FILE)
start = time.time()
dump_svmlight_file(X=test, y=list(range(n_users)), f=TEST_FILE)
logging.debug("Writing testing data took %0.2fs", time.time() - start)

# TODO: rating=1 is not true for negatives, but does not hurt here.
logging.debug("Writing negatives data to file %s", NEGATIVES_FILE)
start = time.time()
dump_svmlight_file(X=negatives, y=list(range(n_users)), f=NEGATIVES_FILE)
logging.debug("Writing negatives took %0.2fs", time.time() - start)

# TODO:
#	* Do not use test data during training
#	* Add option to run on GPU

import argparse
import logging
import numpy as np
import os
import time

from sklearn.datasets import load_svmlight_file
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
# import torch.backends.cudnn as cudnn
# from tensorboardX import SummaryWriter

import model
import evaluate
import data_utils


parser = argparse.ArgumentParser(description='Fit and predict with NCF.')

parser.add_argument('train',
	help='training data file')
parser.add_argument('test',
	help='testing data file')
parser.add_argument('negatives',
	help='negatives data file')
parser.add_argument('ranking'
    , help='output file with 10 ranked samples which contains 9 or 10 negatives'
        + ' and potentially one positive sample.')

parser.add_argument("--model",
	type=str,
	choices=['MLP', 'GMF', 'NeuMF-end', 'NeuMF-pre'],
	default='GMF',
	help="model type")

parser.add_argument("--lr",
	type=float,
	default=0.001,
	help="learning rate")
parser.add_argument("--dropout",
	type=float,
	default=0.0,
	help="dropout rate")
parser.add_argument("--batch_size",
	type=int,
	default=256,
	help="batch size for training")
parser.add_argument("--epochs",
	type=int,
	default=20,
	help="training epoches")
parser.add_argument("--top_k",
	type=int,
	default=10,
	help="compute metrics@top_k")
parser.add_argument("--factor_num",
	type=int,
	default=32,
	help="predictive factors numbers in the model")
parser.add_argument("--num_layers",
	type=int,
	default=3,
	help="number of layers in MLP model")
parser.add_argument("--num_ng",
	type=int,
	default=4,
	help="sample negative items for training")
parser.add_argument("--test_num_ng",
	type=int,
	default=99,
	help="sample part of negative items for testing")
parser.add_argument("--out",
	default=True,
	help="save model or not")
parser.add_argument("--gpu",
	type=str,
	default="0",
	help="gpu card ID")
args = parser.parse_args()

DATA_TYPE = np.float32

MODEL_PATH = './models/'
GMF_MODEL_PATH = MODEL_PATH + 'GMF.pth'
MLP_MODEL_PATH = MODEL_PATH + 'MLP.pth'
#NEUMF_MODEL_PATH = MODEL_PATH + 'NeuMF.pth'

# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# cudnn.benchmark = True

logging.basicConfig(level=logging.DEBUG
    , format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# We load all the three file here to save time in each epoch.
logging.debug("Loading data")
start = time.time()
train, _ = load_svmlight_file(args.train, dtype=DATA_TYPE)
user_num, item_num = train.shape
test, _ = load_svmlight_file(args.test, n_features=item_num, dtype=DATA_TYPE)
negatives, _ = load_svmlight_file(args.negatives, n_features=item_num
    , dtype=DATA_TYPE)
logging.debug("Loading data took %0.2fs", time.time() - start)

logging.debug("Preparing data")
start = time.time()
train_nonzero = train.nonzero()
train_data = [[int(x[0]), int(x[1])] for x in zip(train_nonzero[0]
	, train_nonzero[1])]

train_mat = train.todok()

test_data = []
for u in range(user_num):
	p = test[u].nonzero()
	test_data.append([u, int(p[1])])
	for i in negatives[u].nonzero()[1]:
		test_data.append([u,i])
logging.debug("Preparing data took %0.2fs", time.time() - start)


logging.debug("Constructing train and test datasets")
start = time.time()
train_dataset = data_utils.NCFData(
		train_data, item_num, train_mat, args.num_ng, True)
test_dataset = data_utils.NCFData(
		test_data, item_num, train_mat, 0, False)
train_loader = data.DataLoader(train_dataset,
		batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = data.DataLoader(test_dataset,
		batch_size=args.test_num_ng+1, shuffle=False, num_workers=0)

logging.debug(f"Creating model: {args.model}")
if args.model == 'NeuMF-pre':
	assert os.path.exists(GMF_MODEL_PATH), 'lack of GMF model'
	assert os.path.exists(MLP_MODEL_PATH), 'lack of MLP model'
	GMF_model = torch.load(GMF_MODEL_PATH)
	MLP_model = torch.load(MLP_MODEL_PATH)
else:
	GMF_model = None
	MLP_model = None

model = model.NCF(user_num, item_num, args.factor_num, args.num_layers,
						args.dropout, args.model, GMF_model, MLP_model)
# model.cuda()
loss_function = nn.BCEWithLogitsLoss()

if args.model == 'NeuMF-pre':
	optimizer = optim.SGD(model.parameters(), lr=args.lr)
else:
	optimizer = optim.Adam(model.parameters(), lr=args.lr)

# writer = SummaryWriter() # for visualization

########################## TRAINING #####################################
logging.debug("Learning model")
start = time.time()
count, best_hr = 0, 0
for epoch in range(args.epochs):
	model.train() # Enable dropout (if have).
	start_time = time.time()
	train_loader.dataset.ng_sample()

	for user, item, label in train_loader:

		# user = user.cuda()
		# item = item.cuda()
		# label = label.float().cuda()
		label = label.float()

		model.zero_grad()
		prediction = model(user, item)
		loss = loss_function(prediction, label)
		loss.backward()
		optimizer.step()
		# writer.add_scalar('data/loss', loss.item(), count)
		count += 1

	model.eval()
	HR, NDCG = evaluate.metrics(model, test_loader, args.top_k)

	elapsed_time = time.time() - start_time
	logging.debug("The time elapsed of epoch {:03d}".format(epoch) + " is: " +
			time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
	logging.debug("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

	if HR > best_hr:
		best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
		if args.out:
			if not os.path.exists(MODEL_PATH):
				os.mkdir(MODEL_PATH)
			torch.save(model,
				'{}{}.pth'.format(MODEL_PATH, args.model))

logging.debug("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(
									best_epoch, best_hr, best_ndcg))
logging.debug("Learning model took %0.2fs", time.time() - start)

# loading best model. WARNING: test data is contaminated since it was used for
# model selection.
model = torch.load('{}{}.pth'.format(MODEL_PATH, args.model))

logging.debug("Writing rankings to file %s", args.ranking)
start = time.time()
with open(args.ranking, 'w') as fp:
	for user, item, label in test_loader:
		predictions = model(user, item)
		scores, indices = torch.topk(predictions, 10)
		recommends = torch.take(item, indices).cpu().numpy().tolist()
		fp.write(str(user[0].item()) + ', ')
		fp.write(str(list(zip(recommends, scores.tolist()))))
		fp.write('\n')

logging.debug("Writing rankings took %0.2fs", time.time() - start)

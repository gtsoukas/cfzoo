# Benchmarking of Collaborative Filtering Implementations

 Weighted Regularized Matrix Factorization (WRMF) is used to compress sparse matrices e.g. for [recommender systems](http://yifanhu.net/PUB/cf.pdf). cfzoo is an attempt to compare different implementations of WRMF and other collaborative filtering methods by goodness of fit and compute resource requirements.


## MovieLens dataset w/o hyperparameter tuning
This dataset comes from [grouplens.org](https://grouplens.org/datasets/movielens/).

The following WRMF (hyper)parameters are used where applicable:
* Number of user/item factors: 32
* Regularization: 0.0
* Maximum number of iterations: 3
* Weight (alpha): 1
* Precision: Float 32
* System: 144 cores, 1TB RAM, CPU only

Method   | *n*  | Net Time (sec) | Memory (GiB) | HR@10 (%), 95% CI
---------|------|----------------|--------------|------------------------
pop      | 100K |      0         |   n.a.       | 55.04, (51.86, 58.22)
.        | 1M   |      1         |   0.1        | 55.98, (54.72, 57.23)
.        | 10M  |      7         |   0.3        | 80.45, (80.15, 80.74)
.        | 20M  |      14        |   0.4        | 91.66, (91.52, 91.81)
implicit | 100K |      0         |   n.a.       | 79.85, (77.29, 82.42)
.        | 1M   |      0         |   0.1        | 78.76, (77.73, 79.79)
.        | 10M  |      6         |   0.4        | 90.72, (90.50, 90.93)
.        | 20M  |      15        |   0.7        | 94.43, (94.31, 94.55)
spark    | 100K |      3         |   6.5        | 80.28, (77.73, 82.82)
.        | 1M   |      4         |   7.8        | 80.12, (79.11, 81.12)
.        | 10M  |      4         |   10.4       | 91.73, (91.52, 91.93)
.        | 20M  |      22        |   12.3       | 95.38, (95.27, 95.49)
tf       | 100K |      0         |   0.5        | 78.37, (75.73, 81.00)
.        | 1M   |      2         |   0.6        | 78.56, (77.52, 79.59)
.        | 10M  |      27        |   2.3        | 90.82, (90.60, 91.03)
.        | 20M  |      56        |   4.2        | 94.39, (94.27, 94.51)
lightfm  | 100K |      2         |   0.1        | 81.12, (78.62, 83.63)
.        | 1M   |      16        |   0.1        | 80.61, (79.62, 81.61)
.        | 10M  |      158       |   0.5        | 94.03, (93.85, 94.20)
.        | 20M  |      257       |   0.8        | 97.94, (97.87, 98.02)
ncf      | 100K |      457       |   n.a.       | 82.18, (79.74, 84.63)
.        | 1M   |      9004      |   n.a.       | 83.06, (82.12, 84.01)
.        | 10M  |      450263    |   n.a.       | 92.29, (92.09, 92.48)
.        | 20M  |      n.a       |   n.a.       | n.a.


## MovieLens dataset w/ hyperparameter tuning
Method   | *n*  | Net Time (sec) | Memory (GiB) | HR@10 (%), 95% CI
---------|------|----------------|--------------|------------------------
implicit | 100K |      n.a.      |   0.09       | 79.96, (77.40, 82.52)
.        | 1M   |      n.a.      |   0.15       | 81.90, (80.93, 82.87)
.        | 10M  |      n.a.      |   0.60       | 92.55, (92.36, 92.75)
.        | 20M  |      n.a.      |   1.13       | 96.52, (96.42, 96.62)



## Last.fm dataset w/o hyperparameter tuning
The dataset comes from www.last.fm and was compiled by [Òscar Celma](http://ocelma.net/MusicRecommendationDataset/lastfm-360K.html).

WRMF Parameters are the same as for the non-tuned MovieLens fit.

Method   | *n*  | Net Time (sec) |   Memory (GiB) | HR@10 (%), 95% CI
---------|------|----------------|----------------|------------------------
pop      | 360K |      24        |   0.7          | 88.24, (88.14, 88.35)
implicit | 360K |      20        |   1.0          | 91.71, (91.62, 91.80)
spark    | 360K |      46        |   16.2         | 93.02, (92.93, 93.10)
tf       | 360K |      58        |   3.8          | 91.45, (91.36, 91.54)
lightfm  | 360K |      275       |   1.1          | 97.96, (97.92, 98.01)
ncf      | 360K |      462808    |   n.a.         | 88.23, (88.13, 88.34)


## Methods
* pop -- Popular items
* implicit -- [Ben Fredericksons implicit package 0.3.8 (Python)](https://github.com/benfred/implicit)
* spark -- [Apache Spark 2.4.4 MLlib ALS (JVM)](http://spark.apache.org/)
* tf -- [Google TensorFlow](https://www.tensorflow.org/api_docs/python/tf/contrib/factorization/WALSModel)
* lightfm -- [LightFM (Python)](https://github.com/lyst/lightfm)
* ncf -- Neural Collaborative Filtering (Python) [Paper](https://github.com/guoyang9/NCF), [Torch implementation](https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf). GMF version of model is trained for 20 epochs.


## Evaluation protocol
The evaluation protocol is described [here](https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf). Instead of holding out the most recent interaction for the test set, evaluation is done on exactly one random interaction per user. Note that a large number of non-popular items may not occur in the test set at all.

All input datasets must consist of a two-column dataset where the first column is a user id and the second column is an item id.

The input dataset is split into three datasets:
* train -- a training dataset for model fitting
* test -- for each user one randomly sampled user item interaction
* negatives -- for each user 99 randomly sampled items that the user did not interact with

*Net Time* is the real time used for parameter learning without any data loading, preprocessing or postprocessing.

*Memory* is the *non-swapped physical memory* (RSS) measured via pidstst during model learning and gives a rough approximation of peak memory usage at best.


## Run
All requirements can be installed with the supplied docker file:
```
docker build --rm --tag cfzoo .

docker run \
  -t \
  -i \
  -v ${PWD}:/cfzoo \
  --rm \
  cfzoo

```

Once all required dependencies are installed or from within the docker container, do:
```
./movielens.sh > $(date +%Y%m%d%H%M%S)_movielens.log 2>&1
./movielens_tuned.sh > $(date +%Y%m%d%H%M%S)_movielens_tuned.log 2>&1
./lastfm.sh > $(date +%Y%m%d%H%M%S)_lastfm.log 2>&1
```

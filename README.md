# Benchmarking of Collaborative Filtering Implementations

 Weighted Regularized Matrix Factorization (WRMF) is used to compress sparse matrices e.g. for [recommender systems](http://yifanhu.net/PUB/cf.pdf). cfzoo is an attempt to compare different implementations of WRMF by goodness of fit and other metrics that matter in practice.


## MovieLens dataset w/o hyperparameter tuning
This dataset comes from [grouplens.org](https://grouplens.org/datasets/movielens/). The following WRMF (hyper)parameters are used where applicable:
* Number of user/item factors: 32
* Regularization: 0.0
* Maximum number of iterations: 3
* Weight (alpha): 1
* Precision: Float 32
* System: 144 cores, 1TB RAM

Tool     | *n*  | Net Time (sec) | Memory (MB) | HR@10 (%)
---------|------|----------------|-------------|--------
pop      | 100K |      0         |   n.a.      | 56.73
.        | 1M   |      1         |   n.a.      | 55.58
.        | 10M  |      7         |   n.a.      | 80.74
.        | 20M  |      14        |   n.a.      | 91.85
implicit | 100K |      0         |   n.a.      | 81.55
.        | 1M   |      0         |   n.a.      | 80.07
.        | 10M  |      6         |   n.a.      | 90.68
.        | 20M  |      15        |   n.a.      | 94.60
spark    | 100K |      3         |   n.a.      | 81.34
.        | 1M   |      4         |   n.a.      | 80.84
.        | 10M  |      4         |   n.a.      | 91.57
.        | 20M  |      22        |   n.a.      | 95.44
tf       | 100K |      0         |   n.a.      | 79.11
.        | 1M   |      2         |   n.a.      | 79.88
.        | 10M  |      27        |   n.a.      | 90.67
.        | 20M  |      56        |   n.a.      | 94.33


## Last.fm dataset w/o hyperparameter tuning
The dataset comes from www.last.fm and was compiled by [Òscar Celma](http://ocelma.net/MusicRecommendationDataset/lastfm-360K.html).
Parameters are the same as for the non-tuned MovieLens fit.

Tool     | *n*  | Net Time (sec) | ~ Memory (MB) | HR@10 (%)
---------|------|----------------|---------------|--------
pop      | 360K |      22        |   1           | 88.26
implicit | 360K |      17        |   2           | 91.75
spark    | 360K |      26        |   60          | 93.00
tf       | 360K |      52        |   23          | 91.24


## Tools
* pop -- Popular items
* implicit -- [Ben Fredericksons implicit package 0.3.8 (Python)](https://github.com/benfred/implicit)
* spark -- [Apache Spark 2.4.0 MLlib ALS (JVM)](http://spark.apache.org/)
* tf -- [Google TensorFlow](https://www.tensorflow.org/api_docs/python/tf/contrib/factorization/WALSModel)


## Evaluation protocol
The evaluation protocol is described [here](https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf). Instead of holding out the most recent interaction for the test set, evaluation is done on exactly one random interaction per user. Note that a large number of non-popular items may not occur in the test set at all.

All input datasets must consist of a two-column dataset where the first column is a user id and the second column is an item id.

The input dataset is split into three datasets:
* train -- a training dataset for model fitting
* test -- for each user one randomly sampled user item interaction
* negatives -- for each user 99 randomly sampled items that the user did not interact with

*Net Time* is the real time used for parameter learning without any data loading, preprocessing or postprocessing.

*Memory* is the *virtual memory size* measured via pidstst during model learning and gives a rough approximation at best.


## Run
Once all required dependencies are installed, do:
```
./movielens.sh > $(date +%Y%m%d%H%M%S)_movielens.log 2>&1
./lastfm.sh > $(date +%Y%m%d%H%M%S)_lastfm.log 2>&1
```

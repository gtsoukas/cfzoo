#!/bin/bash

set -e

# https://grouplens.org/datasets/movielens/

# setup python environment
PY_DIR=../../src/main/python
export OPENBLAS_NUM_THREADS=1

# setup Spark environment
mkdir -p tmp
export SPARK_LOCAL_DIRS=./tmp/
SPARK_JAR=../../target/scala-2.11/wrmfzoo_2.11-0.0.1.jar
sbt package


# ml-100k
DATASET=ml-100k
echo ${DATASET}
DATA_DIR=data/${DATASET}
mkdir -p ${DATA_DIR}

cd ${DATA_DIR}
wget --quiet http://files.grouplens.org/datasets/movielens/${DATASET}.zip
unzip ${DATASET}.zip
python ${PY_DIR}/dataprep.py ${DATASET}/u.data

time python ${PY_DIR}/popular.py train.svm test.svm negatives.svm ranking_popular

time python ${PY_DIR}/benfred_implicit.py train.svm test.svm negatives.svm ranking_benfred_implicit

time spark-submit \
  --class "WRMF" \
  --driver-memory 16g \
  ${SPARK_JAR} \
  train.svm \
  test.svm \
  negatives.svm \
  ranking_sparkml \
  >> sparkml.log

time python ${PY_DIR}/google_tf.py train.svm test.svm negatives.svm ranking_google_tf

echo "popular:"
python ${PY_DIR}/evaluator.py train.svm test.svm ranking_popular
echo "benferid_implicit:"
python ${PY_DIR}/evaluator.py train.svm test.svm ranking_benfred_implicit
echo "sparkml:"
python ${PY_DIR}/evaluator.py train.svm test.svm ranking_sparkml
echo "google_tf:"
python ${PY_DIR}/evaluator.py train.svm test.svm ranking_google_tf

cd ../..


# ml-1m
DATASET=ml-1m
echo ${DATASET}
DATA_DIR=data/${DATASET}
mkdir -p ${DATA_DIR}

cd ${DATA_DIR}
wget --quiet http://files.grouplens.org/datasets/movielens/${DATASET}.zip
unzip ${DATASET}.zip

python ${PY_DIR}/dataprep.py ${DATASET}/ratings.dat --separator "::"

time python ${PY_DIR}/popular.py train.svm test.svm negatives.svm ranking_popular

time python ${PY_DIR}/benfred_implicit.py train.svm test.svm negatives.svm ranking_benfred_implicit

time spark-submit \
  --class "WRMF" \
  --driver-memory 16g \
  ${SPARK_JAR} \
  train.svm \
  test.svm \
  negatives.svm \
  ranking_sparkml \
  >> sparkml.log

time python ${PY_DIR}/google_tf.py train.svm test.svm negatives.svm ranking_google_tf

echo "popular:"
python ${PY_DIR}/evaluator.py train.svm test.svm ranking_popular
echo "benferid_implicit:"
python ${PY_DIR}/evaluator.py train.svm test.svm ranking_benfred_implicit
echo "sparkml:"
python ${PY_DIR}/evaluator.py train.svm test.svm ranking_sparkml
echo "google_tf:"
python ${PY_DIR}/evaluator.py train.svm test.svm ranking_google_tf

cd ../..


# ml-10m
DATASET=ml-10m
echo ${DATASET}
DATA_DIR=data/${DATASET}
mkdir -p ${DATA_DIR}

cd ${DATA_DIR}
wget --quiet http://files.grouplens.org/datasets/movielens/${DATASET}.zip
unzip ${DATASET}.zip

python ${PY_DIR}/dataprep.py ml-10M100K/ratings.dat --separator "::"

time python ${PY_DIR}/popular.py train.svm test.svm negatives.svm ranking_popular

time python ${PY_DIR}/benfred_implicit.py train.svm test.svm negatives.svm ranking_benfred_implicit

time spark-submit \
  --class "WRMF" \
  --driver-memory 16g \
  ${SPARK_JAR} \
  train.svm \
  test.svm \
  negatives.svm \
  ranking_sparkml \
  >> sparkml.log

time python ${PY_DIR}/google_tf.py train.svm test.svm negatives.svm ranking_google_tf

echo "popular:"
python ${PY_DIR}/evaluator.py train.svm test.svm ranking_popular
echo "benferid_implicit:"
python ${PY_DIR}/evaluator.py train.svm test.svm ranking_benfred_implicit
echo "sparkml:"
python ${PY_DIR}/evaluator.py train.svm test.svm ranking_sparkml
echo "google_tf:"
python ${PY_DIR}/evaluator.py train.svm test.svm ranking_google_tf

cd ../..



# ml-20m
DATASET=ml-20m
echo ${DATASET}
DATA_DIR=data/${DATASET}
mkdir -p ${DATA_DIR}

cd ${DATA_DIR}
wget --quiet http://files.grouplens.org/datasets/movielens/${DATASET}.zip
unzip ${DATASET}.zip

python ${PY_DIR}/dataprep.py ${DATASET}/ratings.csv --separator "," --skip-headers

time python ${PY_DIR}/popular.py train.svm test.svm negatives.svm ranking_popular

time python ${PY_DIR}/benfred_implicit.py train.svm test.svm negatives.svm ranking_benfred_implicit

time spark-submit \
  --class "WRMF" \
  --driver-memory 16g \
  ${SPARK_JAR} \
  train.svm \
  test.svm \
  negatives.svm \
  ranking_sparkml \
  >> sparkml.log

time python ${PY_DIR}/google_tf.py train.svm test.svm negatives.svm ranking_google_tf

echo "popular:"
python ${PY_DIR}/evaluator.py train.svm test.svm ranking_popular
echo "benferid_implicit:"
python ${PY_DIR}/evaluator.py train.svm test.svm ranking_benfred_implicit
echo "sparkml:"
python ${PY_DIR}/evaluator.py train.svm test.svm ranking_sparkml
echo "google_tf:"
python ${PY_DIR}/evaluator.py train.svm test.svm ranking_google_tf

cd ../..

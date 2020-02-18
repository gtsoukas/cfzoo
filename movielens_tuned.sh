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

# first parameter is the command to log pidstat for
# second parameter is the name of the pidstat logifle to append to
function log_pidstad_cmd {
  CMD=$1
  PIDSTAT_LOG=$2
  $CMD &
  CMD_PID=$!
  pidstat -p $CMD_PID -r 1 >> $PIDSTAT_LOG
  echo "PID of monitord task: ${CMD_PID}"
}


# ml-100k
DATASET=ml-100k
echo ${DATASET}
DATA_DIR=data/${DATASET}
mkdir -p ${DATA_DIR}

cd ${DATA_DIR}
wget --quiet http://files.grouplens.org/datasets/movielens/${DATASET}.zip
unzip ${DATASET}.zip
python ${PY_DIR}/sampler.py ${DATASET}/u.data --validation-set True

log_pidstad_cmd "python ${PY_DIR}/benfred_implicit.py train.svm test.svm negatives.svm ranking_benfred_implicit --validation-set valid.svm" benferid_implicit_pidstat.log
python ${PY_DIR}/parse_pidstat_file.py benferid_implicit_pidstat.log

echo "benferid_implicit:"
python ${PY_DIR}/evaluator.py train.svm test.svm ranking_benfred_implicit

cd ../..


# ml-1m
DATASET=ml-1m
echo ${DATASET}
DATA_DIR=data/${DATASET}
mkdir -p ${DATA_DIR}

cd ${DATA_DIR}
wget --quiet http://files.grouplens.org/datasets/movielens/${DATASET}.zip
unzip ${DATASET}.zip

python ${PY_DIR}/sampler.py ${DATASET}/ratings.dat --separator "::" --validation-set True

log_pidstad_cmd "python ${PY_DIR}/benfred_implicit.py train.svm test.svm negatives.svm ranking_benfred_implicit --validation-set valid.svm" benferid_implicit_pidstat.log
python ${PY_DIR}/parse_pidstat_file.py benferid_implicit_pidstat.log

echo "benferid_implicit:"
python ${PY_DIR}/evaluator.py train.svm test.svm ranking_benfred_implicit

cd ../..


# ml-10m
DATASET=ml-10m
echo ${DATASET}
DATA_DIR=data/${DATASET}
mkdir -p ${DATA_DIR}

cd ${DATA_DIR}
wget --quiet http://files.grouplens.org/datasets/movielens/${DATASET}.zip
unzip ${DATASET}.zip

python ${PY_DIR}/sampler.py ml-10M100K/ratings.dat --separator "::" --validation-set True

log_pidstad_cmd "python ${PY_DIR}/benfred_implicit.py train.svm test.svm negatives.svm ranking_benfred_implicit --validation-set valid.svm" benferid_implicit_pidstat.log
python ${PY_DIR}/parse_pidstat_file.py benferid_implicit_pidstat.log

echo "benferid_implicit:"
python ${PY_DIR}/evaluator.py train.svm test.svm ranking_benfred_implicit

cd ../..



# ml-20m
DATASET=ml-20m
echo ${DATASET}
DATA_DIR=data/${DATASET}
mkdir -p ${DATA_DIR}

cd ${DATA_DIR}
wget --quiet http://files.grouplens.org/datasets/movielens/${DATASET}.zip
unzip ${DATASET}.zip

python ${PY_DIR}/sampler.py ${DATASET}/ratings.csv --separator "," --skip-headers --validation-set True

log_pidstad_cmd "python ${PY_DIR}/benfred_implicit.py train.svm test.svm negatives.svm ranking_benfred_implicit --validation-set valid.svm" benferid_implicit_pidstat.log
python ${PY_DIR}/parse_pidstat_file.py benferid_implicit_pidstat.log

echo "benferid_implicit:"
python ${PY_DIR}/evaluator.py train.svm test.svm ranking_benfred_implicit

cd ../..

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
python ${PY_DIR}/sampler.py ${DATASET}/u.data

log_pidstad_cmd "python ${PY_DIR}/popular.py train.svm test.svm negatives.svm ranking_popular" popular_pidstat.log
python ${PY_DIR}/parse_pidstat_file.py popular_pidstat.log

log_pidstad_cmd "python ${PY_DIR}/benfred_implicit.py train.svm test.svm negatives.svm ranking_benfred_implicit" benferid_implicit_pidstat.log
python ${PY_DIR}/parse_pidstat_file.py benferid_implicit_pidstat.log

log_pidstad_cmd "spark-submit --class WRMF --driver-memory 16g ${SPARK_JAR} train.svm test.svm negatives.svm ranking_sparkml" spark_pidstat.log
python ${PY_DIR}/parse_pidstat_file.py spark_pidstat.log

log_pidstad_cmd "python ${PY_DIR}/google_tf.py train.svm test.svm negatives.svm ranking_google_tf" google_tf_pidstat.log
python ${PY_DIR}/parse_pidstat_file.py google_tf_pidstat.log

log_pidstad_cmd "python ${PY_DIR}/lyst_lightfm.py train.svm test.svm negatives.svm ranking_lyst_lightfm" lyst_lightfm_pidstat.log
python ${PY_DIR}/parse_pidstat_file.py lyst_lightfm_pidstat.log

python ${PY_DIR}/ncf/ncf.py train.svm test.svm negatives.svm ranking_ncf --model=GMF --batch_size=256 --lr=0.001 --factor_num=32 --epochs=20

echo "popular:"
python ${PY_DIR}/evaluator.py train.svm test.svm ranking_popular
echo "benferid_implicit:"
python ${PY_DIR}/evaluator.py train.svm test.svm ranking_benfred_implicit
echo "sparkml:"
python ${PY_DIR}/evaluator.py train.svm test.svm ranking_sparkml
echo "google_tf:"
python ${PY_DIR}/evaluator.py train.svm test.svm ranking_google_tf
echo "lyst_lightfm:"
python ${PY_DIR}/evaluator.py train.svm test.svm ranking_lyst_lightfm
echo "ncf:"
python ${PY_DIR}/evaluator.py train.svm test.svm ranking_ncf

cd ../..


# ml-1m
DATASET=ml-1m
echo ${DATASET}
DATA_DIR=data/${DATASET}
mkdir -p ${DATA_DIR}

cd ${DATA_DIR}
wget --quiet http://files.grouplens.org/datasets/movielens/${DATASET}.zip
unzip ${DATASET}.zip

python ${PY_DIR}/sampler.py ${DATASET}/ratings.dat --separator "::"

log_pidstad_cmd "python ${PY_DIR}/popular.py train.svm test.svm negatives.svm ranking_popular" popular_pidstat.log
python ${PY_DIR}/parse_pidstat_file.py popular_pidstat.log

log_pidstad_cmd "python ${PY_DIR}/benfred_implicit.py train.svm test.svm negatives.svm ranking_benfred_implicit" benferid_implicit_pidstat.log
python ${PY_DIR}/parse_pidstat_file.py benferid_implicit_pidstat.log

log_pidstad_cmd "spark-submit --class WRMF --driver-memory 16g ${SPARK_JAR} train.svm test.svm negatives.svm ranking_sparkml" spark_pidstat.log
python ${PY_DIR}/parse_pidstat_file.py spark_pidstat.log

log_pidstad_cmd "python ${PY_DIR}/google_tf.py train.svm test.svm negatives.svm ranking_google_tf" google_tf_pidstat.log
python ${PY_DIR}/parse_pidstat_file.py google_tf_pidstat.log

log_pidstad_cmd "python ${PY_DIR}/lyst_lightfm.py train.svm test.svm negatives.svm ranking_lyst_lightfm" lyst_lightfm_pidstat.log
python ${PY_DIR}/parse_pidstat_file.py lyst_lightfm_pidstat.log

# python ${PY_DIR}/ncf/ncf.py train.svm test.svm negatives.svm ranking_ncf --model=GMF --batch_size=256 --lr=0.001 --factor_num=32 --epochs=20

echo "popular:"
python ${PY_DIR}/evaluator.py train.svm test.svm ranking_popular
echo "benferid_implicit:"
python ${PY_DIR}/evaluator.py train.svm test.svm ranking_benfred_implicit
echo "sparkml:"
python ${PY_DIR}/evaluator.py train.svm test.svm ranking_sparkml
echo "google_tf:"
python ${PY_DIR}/evaluator.py train.svm test.svm ranking_google_tf
echo "lyst_lightfm:"
python ${PY_DIR}/evaluator.py train.svm test.svm ranking_lyst_lightfm
# echo "ncf:"
# python ${PY_DIR}/evaluator.py train.svm test.svm ranking_ncf

cd ../..


# ml-10m
DATASET=ml-10m
echo ${DATASET}
DATA_DIR=data/${DATASET}
mkdir -p ${DATA_DIR}

cd ${DATA_DIR}
wget --quiet http://files.grouplens.org/datasets/movielens/${DATASET}.zip
unzip ${DATASET}.zip

python ${PY_DIR}/sampler.py ml-10M100K/ratings.dat --separator "::"

log_pidstad_cmd "python ${PY_DIR}/popular.py train.svm test.svm negatives.svm ranking_popular" popular_pidstat.log
python ${PY_DIR}/parse_pidstat_file.py popular_pidstat.log

log_pidstad_cmd "python ${PY_DIR}/benfred_implicit.py train.svm test.svm negatives.svm ranking_benfred_implicit" benferid_implicit_pidstat.log
python ${PY_DIR}/parse_pidstat_file.py benferid_implicit_pidstat.log

log_pidstad_cmd "spark-submit --class WRMF --driver-memory 16g ${SPARK_JAR} train.svm test.svm negatives.svm ranking_sparkml" spark_pidstat.log
python ${PY_DIR}/parse_pidstat_file.py spark_pidstat.log

log_pidstad_cmd "python ${PY_DIR}/google_tf.py train.svm test.svm negatives.svm ranking_google_tf" google_tf_pidstat.log
python ${PY_DIR}/parse_pidstat_file.py google_tf_pidstat.log

log_pidstad_cmd "python ${PY_DIR}/lyst_lightfm.py train.svm test.svm negatives.svm ranking_lyst_lightfm" lyst_lightfm_pidstat.log
python ${PY_DIR}/parse_pidstat_file.py lyst_lightfm_pidstat.log

# python ${PY_DIR}/ncf/ncf.py train.svm test.svm negatives.svm ranking_ncf --model=GMF --batch_size=256 --lr=0.001 --factor_num=32 --epochs=20

echo "popular:"
python ${PY_DIR}/evaluator.py train.svm test.svm ranking_popular
echo "benferid_implicit:"
python ${PY_DIR}/evaluator.py train.svm test.svm ranking_benfred_implicit
echo "sparkml:"
python ${PY_DIR}/evaluator.py train.svm test.svm ranking_sparkml
echo "google_tf:"
python ${PY_DIR}/evaluator.py train.svm test.svm ranking_google_tf
echo "lyst_lightfm:"
python ${PY_DIR}/evaluator.py train.svm test.svm ranking_lyst_lightfm
# echo "ncf:"
# python ${PY_DIR}/evaluator.py train.svm test.svm ranking_ncf

cd ../..



# ml-20m
DATASET=ml-20m
echo ${DATASET}
DATA_DIR=data/${DATASET}
mkdir -p ${DATA_DIR}

cd ${DATA_DIR}
wget --quiet http://files.grouplens.org/datasets/movielens/${DATASET}.zip
unzip ${DATASET}.zip

python ${PY_DIR}/sampler.py ${DATASET}/ratings.csv --separator "," --skip-headers

log_pidstad_cmd "python ${PY_DIR}/popular.py train.svm test.svm negatives.svm ranking_popular" popular_pidstat.log
python ${PY_DIR}/parse_pidstat_file.py popular_pidstat.log

log_pidstad_cmd "python ${PY_DIR}/benfred_implicit.py train.svm test.svm negatives.svm ranking_benfred_implicit" benferid_implicit_pidstat.log
python ${PY_DIR}/parse_pidstat_file.py benferid_implicit_pidstat.log

log_pidstad_cmd "spark-submit --class WRMF --driver-memory 16g ${SPARK_JAR} train.svm test.svm negatives.svm ranking_sparkml" spark_pidstat.log
python ${PY_DIR}/parse_pidstat_file.py spark_pidstat.log

log_pidstad_cmd "python ${PY_DIR}/google_tf.py train.svm test.svm negatives.svm ranking_google_tf" google_tf_pidstat.log
python ${PY_DIR}/parse_pidstat_file.py google_tf_pidstat.log

log_pidstad_cmd "python ${PY_DIR}/lyst_lightfm.py train.svm test.svm negatives.svm ranking_lyst_lightfm" lyst_lightfm_pidstat.log
python ${PY_DIR}/parse_pidstat_file.py lyst_lightfm_pidstat.log

# python ${PY_DIR}/ncf/ncf.py train.svm test.svm negatives.svm ranking_ncf --model=GMF --batch_size=256 --lr=0.001 --factor_num=32 --epochs=20

echo "popular:"
python ${PY_DIR}/evaluator.py train.svm test.svm ranking_popular
echo "benferid_implicit:"
python ${PY_DIR}/evaluator.py train.svm test.svm ranking_benfred_implicit
echo "sparkml:"
python ${PY_DIR}/evaluator.py train.svm test.svm ranking_sparkml
echo "google_tf:"
python ${PY_DIR}/evaluator.py train.svm test.svm ranking_google_tf
echo "lyst_lightfm:"
python ${PY_DIR}/evaluator.py train.svm test.svm ranking_lyst_lightfm
# echo "ncf:"
# python ${PY_DIR}/evaluator.py train.svm test.svm ranking_ncf

cd ../..

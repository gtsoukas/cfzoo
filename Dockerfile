FROM python:3.7-stretch

ENV HADOOP_VERSION=2.7
# ENV SPARK_MAJOR_VERSION=2
ENV SPARK_VERSION=2.4.4
ENV SBT_VERSION=1.3.2

RUN apt-get update \
  && apt-get upgrade -y \
  && apt-get -y install \
    openjdk-8-jdk \
    sysstat \
    unzip \
    wget

# Add Python requirements
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Add Apache Spark
RUN mkdir -p sw \
  && cd sw \
  # Apache Spark
  && wget http://mirror.easyname.ch/apache/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
  && tar xf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz
ENV PATH=/sw/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}/bin/:${PATH}

# Add Scala Build Tool (sbt)
RUN mkdir -p sw \
  && cd sw \
  && wget https://github.com/sbt/sbt/releases/download/v${SBT_VERSION}/sbt-${SBT_VERSION}.tgz \
  && tar xf sbt-${SBT_VERSION}.tgz
  # native linalg support for Apache Spark
  #  && apt-get -y install libgfortran3 libatlas3-base libopenblas-base libatlas-base-dev
  #&& cd /usr/lib/ \
  #&& ln -s liblapack.so.3 liblapack.so \
  #&& ln -s libblas.so.3 libblas.so
  #&& update-alternatives --config libblas.so.3 \
  #&& update-alternatives --config liblapack.so.3
ENV PATH=/sw/sbt/bin/:${PATH}


CMD ["/bin/bash"]

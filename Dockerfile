FROM python:3.8

ENV HADOOP_VERSION 3.2
ENV SPARK_VERSION 3.1.1
ENV SBT_VERSION 1.5.0

RUN apt-get update \
  && apt-get upgrade -y \
  && apt-get -y install \
    openjdk-11-jdk-headless \
    sysstat \
    unzip \
    wget

# Add Python requirements
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Add Apache Spark
RUN cd tmp \
  && wget https://downloads.apache.org/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
  && tar xf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
  && rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
  && mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark
  # native linalg support for Apache Spark
  #  && apt-get -y install libgfortran3 libatlas3-base libopenblas-base libatlas-base-dev
  #&& cd /usr/lib/ \
  #&& ln -s liblapack.so.3 liblapack.so \
  #&& ln -s libblas.so.3 libblas.so
  #&& update-alternatives --config libblas.so.3 \
  #&& update-alternatives --config liblapack.so.3

ENV PATH=/opt/spark/bin/:${PATH}

# Add Scala Build Tool (sbt)
RUN cd tmp \
  && wget https://github.com/sbt/sbt/releases/download/v${SBT_VERSION}/sbt-${SBT_VERSION}.tgz \
  && tar xf sbt-${SBT_VERSION}.tgz \
  && rm sbt-${SBT_VERSION}.tgz \
  && mv sbt /opt/sbt

ENV PATH=/opt/sbt/bin/:${PATH}

RUN mkdir /cfzoo

WORKDIR /cfzoo

CMD ["/bin/bash"]

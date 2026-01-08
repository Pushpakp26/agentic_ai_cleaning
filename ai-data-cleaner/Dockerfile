FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    openjdk-21-jre-headless \
    wget \
    curl \
    procps \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV SPARK_VERSION=4.0.1
ENV HADOOP_VERSION=3
ENV SPARK_HOME=/opt/spark
ENV HADOOP_HOME=/opt/hadoop
ENV PATH=$JAVA_HOME/bin:$SPARK_HOME/bin:$HADOOP_HOME/bin:$PATH
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3
ENV PYTHONPATH=/app

# Install Spark & Hadoop
RUN wget https://downloads.apache.org/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    && tar -xzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    && mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark \
    && rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

RUN wget https://downloads.apache.org/hadoop/common/hadoop-3.3.6/hadoop-3.3.6.tar.gz \
    && tar -xzf hadoop-3.3.6.tar.gz \
    && mv hadoop-3.3.6 /opt/hadoop \
    && rm hadoop-3.3.6.tar.gz

# Copy ROOT requirements.txt (the important one)
COPY ../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project structure
COPY .. .

# Create directories
RUN mkdir -p uploads processed_data logs

EXPOSE 8000

# Use your run_server.py script
CMD ["python", "run_server.py"]

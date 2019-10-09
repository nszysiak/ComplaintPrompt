#!/usr/bin/env python
# coding: utf-8
# @Time    : 2019/10/08 20:40
# @Author  : nszysiak
# @Site    :
# @File    : DataProcessor.py
# @Software: Atom
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.types import *
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.sql.functions import *
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vector
from pyspark.ml.feature import VectorAssembler, StandardScaler
import numpy as np


#def clean_up(line):
parq_wildcard_loc = "preprocessed_complaints\part-*.parquet"

spark_conf = SparkConf()

spark_session = SparkSession.builder \
.master("local[*]") \
.appName("PreliminarySieve") \
.config(conf=spark_conf) \
.getOrCreate()

spark_session.sparkContext.setLogLevel('WARN')

init_df    =   spark_session.read \
                .format("parquet") \
                .option("header", "true") \
                .option("inferSchema", "true") \
                .load(parq_wildcard_loc) \
                .alias("init_df")

feat_cols = init_df.columns

vec_assembler = VectorAssembler(inputCols=feat_cols, outputCol='features')

fd = vec_assembler.transform(init_df)

scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)

scaler_model = scaler.fit(fd)

clstr_final_dt = scaler_model.transform(fd)

#checks

kmeans4 = KMeans(featuresCol="scaledFeatures", k=4)
kmeans3 = KMeans(featuresCol="scaledFeatures", k=3)
kmeans2 = KMeans(featuresCol="scaledFeatures", k=2)

# do a model

model_k4 = kmeans4.fit(clstr_final_dt)
model_k3 = kmeans3.fit(clstr_final_dt)
model_k2 = kmeans2.fit(clstr_final_dt)

wssse_4 = model_k4.computeCost(clstr_final_dt)
wssse_3 = model_k3.computeCost(clstr_final_dt)
wssse_2 = model_k2.computeCost(clstr_final_dt)

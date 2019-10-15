#!/usr/bin/env python
# coding: utf-8
# @Time    : 2019/10/15 23:00
# @Author  : nszysiak
# @Site    :
# @File    : DataProcessor.py
# @Software: Atom
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.column import Column as col
from pyspark.sql.types import *
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.sql.functions import *
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vector
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql.functions import udf
import numpy as np
from pathlib import Path
import re



files_dir = "preprocessed_complaints"
parq_wildcard_loc = "preprocessed_complaints\part-*.parquet"
succ_file_path = "preprocessed_complaints\_SUCCESS"

succ_file = Path(succ_file_path)
file_ex = Path(files_dir)


def cleanse_field(field):
    pattern = r'[^A-Za-z0-9 ]+'
    if field is not None:
        return re.sub(pattern,'',field.lower())
    else:
        return None

def main():

    if file_ex.is_dir() and succ_file.is_file():

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


        udf_cleansed_field = udf(cleanse_field, StringType())

        datatp_chg =  udf (lambda x: datetime.strptime(x, '%m/%d/%Y'), DateType())

        cleansed_init_df = init_df.withColumn('Issue', udf_cleansed_field(init_df['Issue'])) \
                           .withColumn('CompanyResponseToConsument', udf_cleansed_field(init_df['CompanyResponseToConsument'])) \
                           .withColumn('ReceivedDate', datatp_chg(init_df['ReceivedDate']))

        filtered_df = cleansed_init_df.filter(cleansed_init_df['CompanyResponseToConsument'].rlike('close')) \
                      .filter(cleansed_init_df['StateName'] == 'California')

        filtered_df.createOrReplaceTempView("complaints")

        final_df = spark_session.sql("SELECT RowNoIndex, ComplaintId, ReceivedDate, Product, Subproduct, Issue, CompanyResponseToConsument \
                                     FROM complaints")

        final_df.show(20)


        #feat_cols = init_df.columns

        #vec_assembler = VectorAssembler(inputCols=feat_cols, outputCol='features')

        #fd = vec_assembler.transform(init_df).cache()

        #scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)

        #scaler_model = scaler.fit(fd)

        #clstr_final_dt = scaler_model.transform(fd)

        #checks

        #kmeans4 = KMeans(featuresCol="scaledFeatures", k=4)
        #kmeans3 = KMeans(featuresCol="scaledFeatures", k=3)
        #kmeans2 = KMeans(featuresCol="scaledFeatures", k=2)

        # do a model

        #model_k4 = kmeans4.fit(clstr_final_dt)
        #model_k3 = kmeans3.fit(clstr_final_dt)
        #model_k2 = kmeans2.fit(clstr_final_dt)

        #wssse_4 = model_k4.computeCost(clstr_final_dt)
        #wssse_3 = model_k3.computeCost(clstr_final_dt)
        #wssse_2 = model_k2.computeCost(clstr_final_dt)

        #model_k2.clusterCenters.forEach(println)
        #model_k3.clusterCenters.forEach(println)
        #model_k4.clusterCenters.forEach(println)

if __name__ == '__main__':
        main()

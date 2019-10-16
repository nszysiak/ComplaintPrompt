#!/usr/bin/env python
# coding: utf-8
# @Time    : 2019/10/16 20:40
# @Author  : nszysiak
# @Site    :
# @File    : ParquetConverter.py
# @Software: Atom
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.types import *
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.sql.functions import *
from pyspark import SparkFiles


SOURCE_FILE_PATH = "C:/Users/Norbert Szysiak/Desktop/Consumer_Complaints.csv"
AME_STATES = 'AmericanStatesAbb.json'
PARQUET_DIR_MAME = "preprocessed_complaints"

def main():

    spark_conf = SparkConf()

    spark_session = SparkSession.builder \
    .master("local[*]") \
    .appName("ParquetConverter") \
    .config(conf=spark_conf) \
    .getOrCreate()

    spark_session.sparkContext.setLogLevel('WARN')

    customed_schema = StructType([
                    StructField("ReceivedDate", StringType(), True),
                    StructField("Product", StringType(), True),
                    StructField("Subproduct", StringType(), True),
                    StructField("Issue", StringType(), True),
                    StructField("Subissue", StringType(), True),
                    StructField("ConsumerComplaintNarrative", StringType(), True),
                    StructField("CompanyPublicResponse", StringType(), True),
                    StructField("CompanyName", StringType(), True),
                    StructField("State", StringType(), True),
                    StructField("ZipCode", IntegerType(), True),
                    StructField("Tags", StringType(), True),
                    StructField("IsConsumerConsent", StringType(), True),
                    StructField("SubmittedVia", StringType(), True),
                    StructField("SentDate", StringType(), True),
                    StructField("CompanyResponseToConsument", StringType(), True),
                    StructField("IsTimelyResponse", StringType(), True),
                    StructField("IsConsumerDisputed", StringType(), True),
                    StructField("ComplaintId", IntegerType(), True),
                    ])

    complaint_df =   spark_session.read \
                    .format("csv") \
                    .option("header", "true") \
                    .option("delimiter", ",") \
                    .schema(customed_schema) \
                    .option("nullValue", "null") \
                    .option("mode", "DROPMALFORMED") \
                    .load(SOURCE_FILE_PATH) \
                    .alias("complaint_df")

    states_df =   spark_session.read \
                .json(AME_STATES, multiLine=True) \
                .alias("states_df")

    drop_list = ["state", "abbreviation"]

    master_df = complaint_df.join(broadcast(states_df), col("complaint_df.state") == col("states_df.abbreviation"), 'left') \
                .withColumn("RowNoIndex", monotonically_increasing_id()) \
                .withColumnRenamed("name", "StateName") \
                .drop(*drop_list) \
                .select("RowNoIndex","complaint_df.*", "StateName")

   # keep marksuccessfuljobs Hadoop's property as a trigger (_SUCCESS file) to subsquent processing

    master_df.coalesce(1).write \
                        .format("parquet") \
                        .mode("append") \
                        .option("header","true") \
                        .option("mapreduce.fileoutputcommitter.marksuccessfuljobs","true") \
                        .save(PARQUET_DIR_MAME)


    spark_session.stop()

if __name__ == '__main__':
        main()

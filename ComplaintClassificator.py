#!/usr/bin/env python
# coding: utf-8
# @Author  : nszysiak
# @File    : ParquetConverter.py
# @Software: PyCharm

from datetime import datetime
import re

from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.types import *
from pyspark.conf import SparkConf
from pyspark.sql.functions import *
from pyspark.sql.functions import udf

SOURCE_FILE_PATH = "C:/Users/Norbert Szysiak/Desktop/Consumer_Complaints.csv"
AME_STATES = 'AmericanStatesAbb.json'


def cleanse_field(field):
    pattern = r'[^A-Za-z0-9 ]+'
    if field is not None:
        return re.sub(pattern, '', field.lower())
    else:
        return None


def main():
    spark_conf = SparkConf()

    spark_session = SparkSession.builder \
        .master("local[*]") \
        .appName("ParquetConverter") \
        .config(conf=spark_conf) \
        .getOrCreate()

    spark_session.sparkContext.setLogLevel('WARN')

    # create customized schema as a StructType of StructField(s)
    customized_schema = StructType([
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

    # read Consumer_Complaints.csv file and apply schema
    complaint_df = spark_session.read \
        .format("csv") \
        .option("header", "true") \
        .option("delimiter", ",") \
        .schema(customized_schema) \
        .option("nullValue", "null") \
        .option("mode", "DROPMALFORMED") \
        .load(SOURCE_FILE_PATH) \
        .alias("complaint_df")

    # print schema of a complaint_df DataFrame abstraction
    print(complaint_df.schema)

    # some clean-up activities start right here
    # register cleanse_files function as an UDF (UserDefinedFunction)
    udf_cleansed_field = udf(cleanse_field, StringType())

    # provide a lambda function to format date-type field
    change_data_format = udf(lambda x: datetime.strptime(x, '%m/%d/%Y'), DateType())

    # apply predefined udf_cleansed_field function to "Issue" field in order to remove special
    # signs and modify letters to lower case, apply udf_cleansed_field to "CompanyResponseToConsument"
    # field and rename this field to 'CompanyResponse', drop field 'CompanyResponseToConsument',
    # change date format in 'ReceivedDate' field
    cleansed_init_df = complaint_df.withColumn('Issue', udf_cleansed_field(complaint_df['Issue'])) \
        .withColumn('CompanyResponse', udf_cleansed_field(complaint_df['CompanyResponseToConsument'])) \
        .drop('CompanyResponseToConsument') \
        .withColumn('ReceivedDate', change_data_format(complaint_df['ReceivedDate']))

    filtered_response = cleansed_init_df.filter(cleansed_init_df['CompanyResponse'].rlike('close'))

    final_complaint_df = filtered_response.select('ComplaintId', 'ReceivedDate', "State",
                                  'Product', 'Subproduct', 'Issue', 'CompanyResponse') \
        .filter(filtered_response['Issue'].isNotNull()) \
        .filter(filtered_response['Product'].isNotNull()) \
        .orderBy(filtered_response['ReceivedDate'])

    #possible date filtering
    # .filter(year(filtered_response['ReceivedDate']).between(2013, 2015)) \

    states_df = spark_session.read \
        .json(AME_STATES, multiLine=True) \
        .alias("states_df")

    print(states_df.schema)

    drop_list = ["state", "abbreviation"]

    master_df = final_complaint_df.join(broadcast(states_df), col("complaint_df.State") == col("states_df.abbreviation"),
                                  'left') \
        .withColumn("RowNoIndex", monotonically_increasing_id()) \
        .withColumnRenamed("name", "StateName") \
        .drop(*drop_list) \
        .select("RowNoIndex", "complaint_df.*", "StateName")

    master_df.show(20)

    # feat_cols = init_df.columns

    # vec_assembler = VectorAssembler(inputCols=feat_cols, outputCol='features')

    # fd = vec_assembler.transform(init_df).cache()

    # scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)

    # scaler_model = scaler.fit(fd)

    # clstr_final_dt = scaler_model.transform(fd)

    # checks

    # kmeans4 = KMeans(featuresCol="scaledFeatures", k=4)
    # kmeans3 = KMeans(featuresCol="scaledFeatures", k=3)
    # kmeans2 = KMeans(featuresCol="scaledFeatures", k=2)

    # do a model

    # model_k4 = kmeans4.fit(clstr_final_dt)
    # model_k3 = kmeans3.fit(clstr_final_dt)
    # model_k2 = kmeans2.fit(clstr_final_dt)

    # wssse_4 = model_k4.computeCost(clstr_final_dt)
    # wssse_3 = model_k3.computeCost(clstr_final_dt)
    # wssse_2 = model_k2.computeCost(clstr_final_dt)

    # model_k2.clusterCenters.forEach(println)
    # model_k3.clusterCenters.forEach(println)
    # model_k4.clusterCenters.forEach(println)

    spark_session.stop()


if __name__ == '__main__':
    main()

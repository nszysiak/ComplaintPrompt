#!/usr/bin/env python
# coding: utf-8
# @Author  : nszysiak
# @File    : ComplaintClassificator.py
# @Software: PyCharm

from datetime import datetime as dt, timedelta
import re

from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.types import *
from pyspark.conf import SparkConf
from pyspark.sql.functions import *
from pyspark.sql.functions import udf
from pyspark.ml.feature import HashingTF, IDF, Tokenizer

SOURCE_FILE_PATH = "C:/Users/Norbert Szysiak/Desktop/Consumer_Complaints.csv"
AME_STATES = 'AmericanStatesAbb.json'

"""Classify client complaints using both SparkSQL
 and SparkML APIs"""


def cleanse_field(field):
    pattern = r'[^A-Za-z0-9 ]+'
    if field is not None:
        return re.sub(pattern, '', field.lower())
    else:
        return None


def main():
    # instantiate SparkConf
    spark_conf = SparkConf()

    # build SparkSession with already instantiated SparkConf (spark_conf)
    spark_session = SparkSession.builder \
        .master("local[*]") \
        .appName("ComplaintClassificator") \
        .config(conf=spark_conf) \
        .getOrCreate()

    # time of start
    start_time = dt.now()

    # set log level to 'WARN' - reject 'INFO' level logs
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

    # print statistics of a complaint_df DataFrame abstraction
    print("complaint_df has %d records, %d columns." % (complaint_df.count(), len(complaint_df.columns)))
    print("Schema: %s" % complaint_df.columns)

    # some clean-up activities start right here
    # register cleanse_files function as an UDF (UserDefinedFunction)
    udf_cleansed_field = udf(cleanse_field, StringType())

    # provide a lambda function to format date-type field to 'YYYY-MM-DD' pattern
    change_data_format = udf(lambda x: dt.strptime(x, '%m/%d/%Y'), DateType())

    # apply predefined udf_cleansed_field function to "Issue" field in order to remove special
    # signs and modify letters to lower case, apply udf_cleansed_field to "CompanyResponseToConsument"
    # field and rename this field to 'CompanyResponse', drop field 'CompanyResponseToConsument',
    # change date format in 'ReceivedDate' field
    cleansed_init_df = complaint_df.withColumn('Issue', udf_cleansed_field(complaint_df['Issue'])) \
        .withColumn('CompanyResponse', udf_cleansed_field(complaint_df['CompanyResponseToConsument'])) \
        .drop('CompanyResponseToConsument') \
        .withColumn('ReceivedDate', change_data_format(complaint_df['ReceivedDate']))

    # print statistics of a cleansed_init_df DataFrame abstraction
    print("cleansed_init_df has %d records, %d columns." % (cleansed_init_df.count(), len(cleansed_init_df.columns)))
    print("Schema: %s" % cleansed_init_df.columns)

    # apply filter on 'CompanyResponse' field to show only closed complaints
    filtered_response = cleansed_init_df.filter(cleansed_init_df['CompanyResponse'].rlike('close'))

    # print statistics of a filtered_response DataFrame abstraction
    print("filtered_response has %d records, %d columns." % (filtered_response.count(), len(filtered_response.columns)))
    print("Schema: %s" % filtered_response.columns)

    # select a few needed fields, check if some of these fields are not null, so that the data
    # is consistent in further steps of processing, order by 'ReceivedDate' in case of a look-up
    final_complaints = filtered_response.select('ComplaintId', 'ReceivedDate', 'State',
                                                'Product', 'ConsumerComplaintNarrative', 'Issue') \
        .filter(filtered_response['ConsumerComplaintNarrative'].isNotNull()) \
        .filter(filtered_response['Product'].isNotNull()) \
        .orderBy(filtered_response['ReceivedDate'])

    # print statistics of a final_complaints DataFrame abstraction
    print("final_complaints has %d records, %d columns." % (final_complaints.count(), len(final_complaints.columns)))
    print("Schema: %s" % final_complaints.columns)

    # possible filtering on 'ReceivedDate' field for the filtered_response DataFrame abstraction
    # .filter(year(filtered_response['ReceivedDate']).between(2013, 2015)) \

    # read states json provider as a states_df DataFrame abstraction
    states_df = spark_session.read \
        .json(AME_STATES, multiLine=True) \
        .alias("states_df")

    # print statistics of a states_df DataFrame abstraction
    print("states_df has %d records, %d columns." % (states_df.count(), len(states_df.columns)))
    print("Schema: %s" % states_df.columns)

    # list of fields to drop (not needed for the further processing)
    drop_list = ["state", "abbreviation"]

    # left join of final_complaints with states_df on "complaint_df.State" == "states_df.abbreviation"
    # field with explicitly broadcasted states_df DataFrame abstraction in order to reduced costs
    # of communication (keep a read-only dataset cached on each node)
    master_df = final_complaints.join(broadcast(states_df),
                                      col("complaint_df.State") == col("states_df.abbreviation"),
                                      'left') \
        .withColumn("RowNoIndex", monotonically_increasing_id()) \
        .withColumnRenamed("ConsumerComplaintNarrative", "ConsumerComplaint") \
        .withColumnRenamed("name", "StateName") \
        .drop(*drop_list) \
        .select("Product", "ConsumerComplaint")

    # possible filtering on 'State' field for the states_df DataFrame abstraction
    # .where(states_df['name'].contains('California')) \

    # print statistics of a master_df DataFrame abstraction
    print("master_df has %d records, %d columns." % (master_df.count(), len(master_df.columns)))
    print("Schema: %s" % master_df.columns)

    master_df.show(20)

    tokenizer = Tokenizer(inputCol="ConsumerComplaint", outputCol="Words")
    words_data = tokenizer.transform(master_df)

    # each row as a separate document in terms of TF-IDF
    hashing_tf = HashingTF(inputCol="Words", outputCol="RawFeatures", numFeatures=20)
    featurized_data = hashing_tf.transform(words_data)

    idf = IDF(inputCol="RawFeatures", outputCol="Features")
    idfModel = idf.fit(featurized_data)
    rescaled_data = idfModel.transform(featurized_data)

    rescaled_data.select("Product", "Features").show()

    # time of end
    end_time = dt.now()
    print("Elapsed time: %s" % str(end_time - start_time))

    spark_session.stop()


if __name__ == '__main__':
    main()

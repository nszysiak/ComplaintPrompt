from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.types import *
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.sql.functions import *
from os import getcwd


source_file_path = "C:/Users/Norbert Szysiak/Desktop/Consumer_Complaints.csv"
json_broad_file_name = 'AmericanStatesAbb.json'

spark_conf = SparkConf()

spark_session = SparkSession.builder \
        .master("local[*]") \
        .appName("ParquetConverter") \
        .config(conf=spark_conf) \
        .getOrCreate()

spark_session.sparkContext.setLogLevel('ERROR')

customed_schema = StructType([
            StructField("RECEIVED_DATE", StringType(), True),
            StructField("PRODUCT", StringType(), True),
            StructField("SUBPRODUCT", StringType(), True),
            StructField("ISSUE", StringType(), True),
            StructField("SUBISSUE", StringType(), True),
            StructField("CONSUMER_COMPLAINT_NARRATIVE", StringType(), True),
            StructField("COMPANY_PUBLIC_RESPONE", StringType(), True),
            StructField("COMPANY_NAME", StringType(), True),
            StructField("STATE", StringType(), True),
            StructField("ZIP_CODE", IntegerType(), True),
            StructField("TAGS", StringType(), True),
            StructField("IS_CONSUMER_CONSENT", StringType(), True),
            StructField("SUBMITTED_VIA", StringType(), True),
            StructField("SENT_DATE", StringType(), True),
            StructField("COMPANY_RESPONSE_TO_CONSUMENT", StringType(), True),
            StructField("IS_TIMELY_RESPONSE", StringType(), True),
            StructField("IS_CONSUMER_DISPUTED", StringType(), True),
            StructField("COMPLAINT_ID", IntegerType(), True),
            ])

complaint_df =   spark_session.read \
         .format("csv") \
         .option("header", "true") \
         .option("delimiter", ",") \
         .schema(customed_schema) \
         .option("nullValue", "null") \
         .option("mode", "DROPMALFORMED") \
         .load(source_file_path) \
         .alias("complaint_df")

states_df =   spark_session.read \
            .json(json_broad_file_name, multiLine=True) \
            .alias("states_df")

drop_list = ["state", "abbreviation"]

master_df = complaint_df.join(broadcast(states_df), col("complaint_df.state") == col("states_df.abbreviation"), 'left') \
                        .withColumn("INDEX", monotonically_increasing_id()) \
                        .withColumnRenamed("name", "STATE_NAME") \
                        .drop(*drop_list) \
                        .drop_duplicates() \
                        .select("INDEX","complaint_df.*", "STATE_NAME")

#.sql("SELECT /*+ BROADCAST(states) */ * FROM complaints as c LEFT OUTER JOIN states s ON c.state = s.abbreviation") \

#master_df.show()
#master_df.printSchema()

parquet_dir_name = "preprocessed_complaints"

master_df.coalesce(1).write.format("parquet").mode("append").save(parquet_dir_name)

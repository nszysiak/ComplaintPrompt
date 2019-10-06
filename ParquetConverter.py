from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.types import StructType, StructField, DateType, StringType, IntegerType
from pyspark.conf import SparkConf
from pyspark.context import SparkContext

source_file_path = "C:/Users/Norbert Szysiak/Desktop/Consumer_Complaints.csv"
json_broad_file_name = 'AmericanStatesAbb.json'

spark_conf = SparkConf()

spark = SparkSession.builder \
        .master("local[*]") \
        .appName("ParquetConverter") \
        .config(conf=spark_conf) \
        .getOrCreate()

spark.sparkContext.setLogLevel('ERROR')

schema = StructType([
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

df =   spark.read \
         .format("csv") \
         .option("header", "true") \
         .option("delimiter", ",") \
         .schema(schema) \
         .option("nullValue", "null") \
         .option("mode", "DROPMALFORMED") \
         .load(source_file_path)

broadcasted_var =   spark.read.json(json_broad_file_name)


broadcasted_var.show()
#print(type(df))
# df.printSchema()

#!/usr/bin/env python
# coding: utf-8
# @Author  : nszysiak
# @File    : ComplaintClassificator.py
# @Software: PyCharm
# @Framework: Apache Spark 2.4.4

from datetime import datetime as dt
import re
import sys

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.context import SparkContext
from py4j.protocol import Py4JJavaError
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.types import *
from pyspark.conf import SparkConf
from pyspark.sql.functions import *
from pyspark.sql.functions import udf
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes, NaiveBayesModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

CONSUMER_COMPLAINTS = sys.argv[1]  # "C:/Users/Norbert Szysiak/Desktop/Consumer_Complaints.csv"
AMERICAN_STATES = sys.argv[2]  # "AmericanStatesAbb.json"
AWS_ACCESS_KEY_ID = None
AWS_SECRET_ACCESS_KEY = None

try:
    AWS_ACCESS_KEY_ID = sys.argv[3]
except IndexError:
    pass
try:
    AWS_SECRET_ACCESS_KEY = sys.argv[4]
except IndexError:
    pass


def is_not_blank(_str):
    if _str and _str.strip():
        return True
    return False


def cleanse_field(field):
    pattern = r'[^A-Za-z0-9 ]+'
    if field is not None:
        return re.sub(pattern, '', field.lower())
    else:
        return None


def main():
    # Instantiate SparkConf and sent extraJavaOptions to both executors and drivers
    spark_conf = (SparkConf().set('spark.executor.extraJavaOptions', '-Dcom.amazonaws.services.s3.enableV4=true')
                             .set('spark.driver.extraJavaOptions', '-Dcom.amazonaws.services.s3.enableV4=true'))

    # Instantiate SparkContext based on SparkConf
    sc = SparkContext(conf=spark_conf)

    # Set enableV4 property to access S3 input data
    sc.setSystemProperty('com.amazonaws.services.s3.enableV4', 'true')

    # Create new Hadoop Configuration
    hadoopConf = sc._jsc.hadoopConfiguration()

    # Set Hadoop configuration K-V
    if is_not_blank(AWS_ACCESS_KEY_ID):
        hadoopConf.set('fs.s3a.awsAccessKeyId', AWS_ACCESS_KEY_ID)
    if is_not_blank(AWS_SECRET_ACCESS_KEY):
        hadoopConf.set('fs.s3a.awsSecretAccessKey', AWS_SECRET_ACCESS_KEY)
    hadoopConf.set('com.amazonaws.services.s3a.enableV4', 'true')
    hadoopConf.set('fs.s3a.impl', 'org.apache.hadoop.fs.s3a.S3AFileSystem')

    # Create SparkSession from SparkContext
    spark_session = (SparkSession(sc).builder
                     .appName('ComplaintClassificator')
                     .config(conf=spark_conf)
                     .getOrCreate())

    # Timestamp of start
    start_timestamp = dt.now()

    # Instantiate SparkContext
    sc = spark_session.sparkContext

    # Instantiate SQLContext
    sql_ctx = SQLContext(sc)

    # Set log level to 'WARN'
    sc.setLogLevel('WARN')

    # Set up log4j logging
    log4j_logger = sc._jvm.org.apache.log4j
    logger = log4j_logger.LogManager.getLogger(__name__)

    # Create schema as a StructType of StructField(s)
    schema = StructType([
        StructField('ReceivedDate', StringType(), True),
        StructField('Product', StringType(), True),
        StructField('Subproduct', StringType(), True),
        StructField('Issue', StringType(), True),
        StructField('Subissue', StringType(), True),
        StructField('ConsumerComplaintNarrative', StringType(), True),
        StructField('CompanyPublicResponse', StringType(), True),
        StructField('CompanyName', StringType(), True),
        StructField('State', StringType(), True),
        StructField('ZipCode', IntegerType(), True),
        StructField('Tags', StringType(), True),
        StructField('IsConsumerConsent', StringType(), True),
        StructField('SubmittedVia', StringType(), True),
        StructField('SentDate', StringType(), True),
        StructField('CompanyResponseToConsument', StringType(), True),
        StructField('IsTimelyResponse', StringType(), True),
        StructField('IsConsumerDisputed', StringType(), True),
        StructField('ComplaintId', IntegerType(), True)])

    logger.warn("Starting preprocessing and data cleansing...")

    # Read Consumer_Complaints.csv file and apply schema
    complaint_df = (spark_session.read
                    .format('csv')
                    .option('header', 'true')
                    .option('delimiter', ',')
                    .option('mode', 'FAILFAST')
                    .option('parserLib', 'univocity')
                    .option('escape', '"')
                    .option('multiLine', 'true')
                    .option('inferSchema', 'false')
                    .schema(schema)
                    .load(CONSUMER_COMPLAINTS)
                    .alias('complaint_df'))

    logger.warn("Explaining complaint_df...")
    complaint_df.explain()

    logger.warn("complaint_df has %d records, %d columns." % (complaint_df.count(), len(complaint_df.columns)))
    logger.warn("Printing schema of complaint_df: ")
    complaint_df.printSchema()

    # Register cleanse_files function as an UDF (UserDefinedFunction)
    udf_cleansed_field = udf(cleanse_field, StringType())

    # Provide a lambda function to format date-type field to 'YYYY-MM-DD' pattern
    change_data_format = udf(lambda x: dt.strptime(x, '%m/%d/%Y'), DateType())

    # Do some clean-up activities
    cleansed_df = (complaint_df.withColumn('Issue', udf_cleansed_field(complaint_df['ConsumerComplaintNarrative']))
                   .withColumn('ReceivedDate', change_data_format(complaint_df['ReceivedDate'])))

    logger.warn("Explaining cleansed_df...")
    cleansed_df.explain()

    logger.warn("cleansed_init_df has %d records, %d columns." % (cleansed_df.count(), len(cleansed_df.columns)))
    logger.warn("Printing schema of cleansed_df: ")
    cleansed_df.printSchema()

    # Reduce a number of fields and filter non-null values out on consumer complaint narratives
    final_complaints_df = (cleansed_df.where(cleansed_df['ConsumerComplaintNarrative'].isNotNull())
                           .select('ComplaintId', 'ReceivedDate', 'State', 'Product',
                                   'ConsumerComplaintNarrative', 'Issue')
                           .orderBy(cleansed_df['ReceivedDate']))

    final_complaints_df.registerTempTable("final_complaints_df")

    # Check random ConsumerComplaintNarrative as well as Issue content
    sql_ctx.sql(""" SELECT RowNum, ConsumerComplaintNarrative, Issue FROM
                    (SELECT ROW_NUMBER() OVER (PARTITION BY State ORDER BY ReceivedDate DESC) AS RowNum,
                        ConsumerComplaintNarrative,
                        Issue,
                        ReceivedDate,
                        State
                    FROM final_complaints_df) fc
                    WHERE RowNum = 1
                    LIMIT 10
                    """).show()

    logger.warn("Explaining final_complaints_df...")
    final_complaints_df.explain()

    logger.warn("final_complaints has %d records, %d columns." %
                (final_complaints_df.count(), len(final_complaints_df.columns)))
    logger.warn("Printing schema of final_complaints_df: ")
    final_complaints_df.printSchema()

    # Read states json provider as a states_df DataFrame abstraction
    states_df = (spark_session.read
                 .json(AMERICAN_STATES, multiLine=True)
                 .alias('states_df'))

    logger.warn("Explaining states_df...")
    states_df.explain()

    logger.warn("states_df has %d records, %d columns." % (states_df.count(), len(states_df.columns)))
    logger.warn("Printing schema of states_df: ")
    states_df.printSchema()

    # List of fields to drop (not needed for the further processing)
    drop_list = ['state', 'abbreviation']

    # Join complaints data with American states, apply id field and drop unnecessary fields
    joined_df = (
        final_complaints_df.join(broadcast(states_df), col('complaint_df.State') == col('states_df.abbreviation'), "left")
        .withColumnRenamed('ConsumerComplaintNarrative', 'ConsumerComplaint')
        .withColumn('RowNoIndex', monotonically_increasing_id())
        .select('Product', 'ConsumerComplaint', 'name')
        .drop(*drop_list))

    joined_df.registerTempTable("joined_df")

    # Check random FullStateName content
    sql_ctx.sql(""" SELECT RowNum, Product, ConsumerComplaint, FullStateName FROM
                        (SELECT ROW_NUMBER() OVER (PARTITION BY Product ORDER BY ConsumerComplaint DESC) AS RowNum,
                            Product,
                            ConsumerComplaint,
                            name AS FullStateName
                        FROM joined_df) jd
                        WHERE RowNum = 1
                        LIMIT 10
                        """).show()

    logger.warn("Explaining joined_df...")
    joined_df.explain()

    logger.warn("joined_df has %d records, %d columns." % (joined_df.count(), len(joined_df.columns)))
    logger.warn("Printing schema of joined_df: ")
    joined_df.printSchema()

    # Check unique labels of Product attribute before replace
    joined_df.select('Product').distinct().show()

    # Replace redundant labels from Product field
    renamed_df = (joined_df.withColumn('Product', regexp_replace('Product', "Credit reporting, credit repair services, or other personal consumer reports", "Credit reporting, repair, or other"))
                  .withColumn('Product', regexp_replace("Product", "Virtual currency", "Money transfer, virtual currency, or money service"))
                  .withColumn('Product', regexp_replace("Product", "Money transfer", "Money transfer, virtual currency, or money service"))
                  .withColumn('Product', regexp_replace("Product", "Payday loan", "Payday loan, title loan, or personal loan"))
                  .withColumn('Product', regexp_replace("Product", "Credit reporting", "Credit reporting, repair, or other"))
                  .withColumn('Product', regexp_replace("Product", "Prepaid card", "Credit card or prepaid card"))
                  .withColumn('Product', regexp_replace("Product", "Credit card", "Credit card or prepaid card")))

    renamed_df.registerTempTable("renamed_df")

    # Check how many unique labels (classes) there are
    sql_ctx.sql(""" SELECT DISTINCT Product FROM renamed_df """).show()

    # Check how many times each class occurs in the corpus
    sql_ctx.sql(""" SELECT Product, count(*) 
    FROM renamed_df GROUP BY Product 
    ORDER BY count(*) DESC""").show(50, False)

    logger.warn("Explaining renamed_df...")
    renamed_df.explain()

    # Check unique labels of Product attribute after replace
    renamed_df.select('Product').distinct().show()

    # Check amount of unique labels of Product attribute after replace
    logger.warn(str(renamed_df.select('Product').distinct().count()))

    logger.warn("Starting feature extraction...")

    # Tokenize consumer complaints sentences
    tokenizer = Tokenizer(inputCol='ConsumerComplaint', outputCol='Words')

    # Remove stop words
    remover = StopWordsRemover(inputCol='Words', outputCol='FilteredWords')

    # num_features = 700
    hashing_tf = HashingTF(inputCol='FilteredWords', outputCol='RawFeatures')

    # minDocFreq: minimum number of documents in which a term should appear for filtering
    idf = IDF(inputCol='RawFeatures', outputCol='features')

    # Instantiate StringIndexer
    product_indexer = StringIndexer(inputCol='Product', outputCol='label')

    # Create a pipeline from previously defined feature extraction stages
    pipeline = Pipeline(stages=[tokenizer, remover, hashing_tf, idf, product_indexer])

    # Fit renamed_df to the pipeline
    pipeline_fit = pipeline.fit(renamed_df)

    # Transform pipeline_fit
    data_set = pipeline_fit.transform(renamed_df)

    # Randomly slice the data into training and test datasets with requested ratio
    (training_data, test_data) = data_set.randomSplit([0.7, 0.3], seed=100)

    # Cache training_data
    training_data.cache()

    logger.warn("Starting Naive-Bayes...")

    # Naive-Bayes
    nb = NaiveBayes(labelCol='label', featuresCol='features', modelType='multinomial')

    # Create a model without Cross Validation
    nb_model = nb.fit(training_data)

    # Make predictions on model without Cross Validation
    predictions = nb_model.transform(test_data)

    print("NaiveBayes without CV model type: ", nb.getModelType())
    print("NaiveBayes without CV smoothing factor: ", str(nb.getSmoothing()))

    # NB without CV metrics
    nb_metrics_rdd = MulticlassMetrics(predictions['label', 'prediction'].rdd)

    # NB stats by each class (label)
    labels = predictions.rdd.map(lambda cols: cols.label).distinct().collect()

    logger.warn("Printing NB stats...")

    for label in sorted(labels):
        try:
            print("Class %s precision = %s" % (label, nb_metrics_rdd.precision(label)))
            print("Class %s recall = %s" % (label, nb_metrics_rdd.recall(label)))
            print("Class %s F1 Measure = %s" % (label, nb_metrics_rdd.fMeasure(label, beta=1.0)))
        except Py4JJavaError:
            pass

    # Weighted stats
    print("Weighted recall = %s" % nb_metrics_rdd.weightedRecall)
    print("Weighted precision = %s" % nb_metrics_rdd.weightedPrecision)
    print("Weighted F(1) Score = %s" % nb_metrics_rdd.weightedFMeasure())
    print("Weighted F(0.5) Score = %s" % nb_metrics_rdd.weightedFMeasure(beta=0.5))
    print("Weighted false positive rate = %s" % nb_metrics_rdd.weightedFalsePositiveRate)

    # Show 10 results of predictions that haven't been predicted successfully
    predictions.filter(predictions['prediction'] != predictions['label']) \
        .select("Product", "ConsumerComplaint", "probability", "label", "prediction") \
        .orderBy("probability", ascending=False) \
        .show(n=10, truncate=20)

    # Show 10 results of predictions that have been predicted successfully
    predictions.filter(predictions['prediction'] == predictions['label']) \
        .select("Product", "ConsumerComplaint", "probability", "label", "prediction") \
        .orderBy("probability", ascending=False) \
        .show(n=10, truncate=20)

    # Instantiate an evaluation of predictions without Cross Validation
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

    # Evaluate best model without an use of Cross Validation
    accuracy_without_cv = evaluator.evaluate(predictions)

    print("Naive-Bayes accuracy without Cross Validation = %s (metric)" % str(nb_metrics_rdd.accuracy))

    logger.warn("Starting Cross Validation...")

    # Instantiate ParamGridBuilder for the Cross Validation purpose
    nbp_params_grid = (ParamGridBuilder()
                       .addGrid(nb.smoothing, [0.8, 0.9, 1.0])
                       .addGrid(hashing_tf.numFeatures, [700, 720])
                       .addGrid(idf.minDocFreq, [3, 4, 5])
                       .build())

    # Instantiate the Evaluator of the model
    nb_evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction')

    # Instantiate 5-fold CrossValidator
    nb_cv = CrossValidator(estimator=nb,
                           estimatorParamMaps=nbp_params_grid,
                           evaluator=nb_evaluator,
                           numFolds=5)

    # Create a model with Cross Validation
    nb_cv_model = nb_cv.fit(training_data)

    # Make predictions on model with Cross Validation
    cv_predictions = nb_cv_model.transform(training_data)

    # Evaluate best model with an use of Cross Validation
    accuracy_with_cv = nb_evaluator.evaluate(cv_predictions)

    print("Naive-Bayes accuracy with Cross Validation:", str(accuracy_with_cv))

    print("Improvement for the best fitted model (NB with CV) in regard of NB: ",
          str(accuracy_with_cv - nb_metrics_rdd.accuracy))

    # NB with CV metrics
    nb_with_cv_metrics_rdd = MulticlassMetrics(cv_predictions['label', 'prediction'].rdd)

    # NB with CV stats by each class (label)
    labels = cv_predictions.rdd.map(lambda att: att.label).distinct().collect()

    logger.warn("Printing NB stats...")

    for label in sorted(labels):
        try:
            print("Class %s precision = %s" % (label, nb_with_cv_metrics_rdd.precision(label)))
            print("Class %s recall = %s" % (label, nb_with_cv_metrics_rdd.recall(label)))
            print("Class %s F1 Measure = %s" % (label, nb_with_cv_metrics_rdd.fMeasure(label, beta=1.0)))
        except Py4JJavaError:
            pass

    # Print weighted stats
    print("Weighted recall = %s" % nb_with_cv_metrics_rdd.weightedRecall)
    print("Weighted precision = %s" % nb_with_cv_metrics_rdd.weightedPrecision)
    print("Weighted F(1) Score = %s" % nb_with_cv_metrics_rdd.weightedFMeasure())
    print("Weighted F(0.5) Score = %s" % nb_with_cv_metrics_rdd.weightedFMeasure(beta=0.5))
    print("Weighted false positive rate = %s" % nb_with_cv_metrics_rdd.weightedFalsePositiveRate)

    # Show 10 results of cv_predictions that have been predicted successfully
    (cv_predictions.filter(cv_predictions['prediction'] == cv_predictions['label'])
     .select('Product', 'ConsumerComplaint', 'probability', 'label', 'prediction')
     .orderBy('probability', ascending=False)
     .show(n=10, truncate=20))

    # Show 10 results of cv_predictions that haven't been predicted successfully
    (cv_predictions.filter(cv_predictions['prediction'] != cv_predictions['label'])
     .select('Product', 'ConsumerComplaint', 'probability', 'label', 'prediction')
     .orderBy('probability', ascending=False)
     .show(n=10, truncate=20))

    # Timestamp of end
    end_timestamp = dt.now()

    # Print elapsed time
    print("Elapsed time: %s" % str(end_timestamp - start_timestamp))

    # Stop SparkSession
    spark_session.stop()


if __name__ == '__main__':
    main()

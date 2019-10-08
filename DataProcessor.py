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
from pyspark import SparkFiles

class CleanerUp(object):


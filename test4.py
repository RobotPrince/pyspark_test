# from pyspark.sql.pandas.functions import pandas_udf
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql import  functions as F
import pandas as pd
import numpy as np
# from pyspark.sql import functions as F, pandas as pd
# from pyspark.sql.types import *
from pyspark.sql import SparkSession
# spark = SparkSession.builder.getOrCreate()

# schema = StructType().add('name', StringType(), True).add('amount', DoubleType(), True).add('id', StringType(),True)
# df = spark.read.schema(schema).option("header", True).csv("./input.csv")
# df2 = spark.read.option("header", True).csv("./input2.csv")
# df = spark.read.schema(schema).csv("./input.csv")
# df3 = spark.read.csv("./input3.csv")
# df_join = df.join(df2, 'name', 'left').select(df.name,
#                                               df['amount'].alias('amount1'),
#                                               df2['amount'].alias('amount2'),
#                                               (df['amount']/df2['amount']).alias('amount3'))
# df.printSchema()
# df_csv = df.select("name","id",df['amount'].cast(DecimalType(20,9)).alias("c3"))
# df_csv = df_join.select("name", F.round('amount1', 9), F.round('amount2', 9).alias('amount2'),F.round('amount3', 9).alias('amount3'))
#
# pdf = df_csv.toPandas()
# pdf.to_csv("pdf.csv")

data = pd.read_csv("./input.csv")
data = data.iloc[:, 0:1]  # 按位置取某几列
data2 = data.loc[:, ['name']]
data3 = np.array(data2)
print(type(data3))
data_list = data3.tolist()
print(type(data_list))
# print(data)
# print(data2)
for i in data_list:
    print(i)
# print(df.schema)
# df.select('name', (df['amount'] + 1).alias("amt")).show()
# df.join(df2, "id").select(df["name"], df2["amount"] + df2["amount"]).show()
# df.select('name', F.when(df.amount > 2000, 0).otherwise(df["amount"])).show()
# df.select('name', F.when(df.amount > 2000, 0).otherwise(100).alias('amount2')).show()
# df.createOrReplaceTempView("tableA")
# print("**************")
# @pandas_udf("double")
# def add_one(s: pd.Series) -> pd.Series:
#     return s + 1
#
#
# # spark.udf.register("add_one", add_one)
# spark.sql("select add_one(amount) from tableA").show()


# df_sum = df.groupBy("name").agg(sum("amount"))
# df3_sum = df3.groupBy("name").agg(sum("amount")).alias("s_amount")
# df3_agg = df3.groupBy("name").sum("amount")
# df3_agg.show()
# df3_sum = df3.groupBy("name").count()
# df3_sum = df3.agg(count("amount"))
# df3_sum.show()
#
# df_ts = spark.createDataFrame(
#     [(1, 1.0, 100.0, 11), (1, 2.0, 200.0, 12), (2, 3.0, 300.0, 2), (2, 5.0, 500.0, 2), (2, 10.0, 1000.0, 2)],("id", "v", "v_2", "id_2"))

#
# @pandas_udf("double", PandasUDFType.GROUPED_AGG)
# def mean_udf(v):
#      return v.mean()
#
#
# @pandas_udf("double", PandasUDFType.GROUPED_AGG)
# def max_udf_add_one(v_2):
#     return v_2.max() + 100000



# df_ts.groupby("id", "id_2").agg(mean_udf(df_ts['v']), max_udf_add_one(df_ts["v_2"])).show()
#
# df5= spark.createDataFrame([
#     ['red', 'banana', 1, 10], ['blue', 'banana', 2, 20], ['red', 'carrot', 3, 30],
#     ['blue', 'grape', 4, 40], ['red', 'carrot', 5, 50], ['black', 'carrot', 6, 60],
#     ['red', 'banana', 7, 70], ['red', 'grape', 8, 80]], schema=['color', 'fruit', 'v1', 'v2'])
# df5.show()
#
#
# def plus_mean(pandas_df):
#     return pandas_df.assign(v1=pandas_df.v1 - pandas_df.v1.mean())
#
# df5.groupBy('color').avg().show()
# df5.groupby('color').applyInPandas(plus_mean, schema=df5.schema).show()

from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import mlflow
from modelclass import PySparkLinearRegressionModel
from mlflow.models.signature import infer_signature
import os
import pandas as pd
from mlflow.types.schema import Schema, ColSpec
# Define a predict function that applies the Spark model to input data
def predict(context,spark_model, model_input):
    spark = context.spark_session
    data = spark.createDataFrame(model_input)
    predictions = spark_model.transform(data)
    return predictions.select("prediction").toPandas().values
    # Define a function that returns the model signature
def schema_signature():
    input_schema = pd.DataFrame(columns=["Age", "Tonnage", "passengers", "length", "cabins", "passenger_density", "cruise_cat"])
    output_schema = pd.DataFrame(columns=["prediction"])
    return infer_signature(input_schema, output_schema)
# Initialize an active SparkSession
spark = SparkSession.builder.appName("housing_price_model").getOrCreate()
# Load the input CSV file into a Spark DataFrame
df = spark.read.csv("cruise_ship_info.csv", inferSchema=True, header=True)

# Create a StringIndexer to convert the categorical feature "Cruise_line" to a numerical feature
indexer = StringIndexer(inputCol="Cruise_line", outputCol="cruise_cat")
indexed = indexer.fit(df).transform(df)

# Create a VectorAssembler to combine all the input features into a single vector "features"
assembler = VectorAssembler(
    inputCols=[
        "Age",
        "Tonnage",
        "passengers",
        "length",
        "cabins",
        "passenger_density",
        "cruise_cat",
    ],
    outputCol="features",
)
output = assembler.transform(indexed)

# Split the data into training and test sets
train_data, test_data = output.randomSplit([0.7, 0.3])

# Create a LinearRegression model and train it on the training data
ship_lr = LinearRegression(featuresCol="features", labelCol="crew")
trained_ship_model = ship_lr.fit(train_data)

# Evaluate the trained model and log the R2 score and model type with MLflow
ship_results = trained_ship_model.evaluate(train_data)

with mlflow.start_run(run_name="linear_regression"):
    # Log the training metrics and parameters with MLflow
    mlflow.log_metric("r2_score", ship_results.r2)
    mlflow.set_tag("model_type", "LinearRegression")
    
    # Register the trained model in the MLflow Model Registry
    py_model = PySparkLinearRegressionModel()
    artifact_path="model_87"
    mlflow.spark.log_model(trained_ship_model, artifact_path, conda_env=None, code_paths=None, dfs_tmpdir=None, sample_input=None, registered_model_name='model_87',signature=schema_signature())
    unlabeled_data = test_data.select("features")
    predictions = trained_ship_model.transform(unlabeled_data)
    print(predictions.show())
    
print('Model is registered')
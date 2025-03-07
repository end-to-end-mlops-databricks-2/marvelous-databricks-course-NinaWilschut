# Databricks notebook source
# MAGIC %pip install power_consumption-0.0.1-py3-none-any.whl


# COMMAND ----------
# MAGIC %restart_python

# COMMAND ----------
import os
import time
from pathlib import Path
from typing import Dict, List

import requests
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from src.power_consumption.config import ProjectConfig
from src.power_consumption.serving.model_serving import ModelServing

# spark session

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

# get environment variables
os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")

# Load project config
root = Path(__file__).parent.parent
config = ProjectConfig.from_yaml(config_path=root / "project_config.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name

# Initialize feature store manager
model_serving = ModelServing(
    model_name=f"{catalog_name}.{schema_name}.power_consumption_model", endpoint_name="power-consumption-model-serving"
)

# Deploy the model serving endpoint
model_serving.deploy_or_update_serving_endpoint()

# Create a sample request body
required_columns = [
    "datetime",
    "temperature",
    "humidity",
    "wind_speed",
    "general_diffuse_flows",
    "diffuse_flows",
]

# Sample 1000 records from the training set
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").toPandas()

# Sample 100 records from the training set
sampled_records = test_set[required_columns].sample(n=100, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]


# Call the endpoint with one sample record
def call_endpoint(self, record: List(Dict)):
    """
    Calls the model serving endpoint with a given input record.
    """
    serving_endpoint = f"https://{os.environ['DBR_HOST']}/serving-endpoints/{self.endpoint_name}/invocations"

    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": record},
    )
    return response.status_code, response.text


status_code, response_text = call_endpoint(dataframe_records[0])
print(f"Response Status: {status_code}")
print(f"Response Text: {response_text}")

# "load test"

for i in range(len(dataframe_records)):
    call_endpoint(dataframe_records[i])
    time.sleep(0.2)

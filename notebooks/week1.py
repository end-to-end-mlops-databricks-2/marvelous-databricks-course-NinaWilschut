import logging

import pandas as pd
import yaml
from pyspark.sql import SparkSession

# from src.power_consumption.config import ProjectConfig


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load configuration
with open("../project_config.yml", "r") as file:
    config = yaml.safe_load(file)

catalog_name = config["catalog_name"]
schema_name = config["schema_name"]

spark = SparkSession.builder.getOrCreate()

# Only works in a Databricks environment if the data is there
# to put data there, create volume and run databricks fs cp <path> dbfs:/Volumes/mlops_dev/<schema_name>/<volume_name>/
filepath = f"/Volumes/{catalog_name}/{schema_name}/data/data.csv"
# Load the data
# df = pd.read_csv(filepath)


# Works both locally and in a Databricks environment
filepath = "../data/data.csv"
# Load the data
df = pd.read_csv(filepath)

# check types of columns
print(df.dtypes)

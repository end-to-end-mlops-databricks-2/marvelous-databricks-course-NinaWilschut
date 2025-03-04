import argparse
import logging as logger

import yaml
from pyspark.sql import SparkSession

from src.power_consumption.config import ProjectConfig
from src.power_consumption.data_processor import DataProcessor, generate_synthetic_data

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--env",
    action="store",
    default=None,
    type=str,
    required=True,
)

args = parser.parse_args()
root_path = args.root_path
config_path = f"{root_path}/files/project_config.yml"

config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# Load the house prices dataset
spark = SparkSession.builder.getOrCreate()

df = spark.read.csv(
    f"/Volumes/{config.catalog_name}/{config.schema_name}/data/data.csv", header=True, inferSchema=True
).toPandas()

logger.info(f"{df.shape[0]} rows and {df.shape[1]} columns loaded.")

# Generate synthetic data
### This is mimicking a new data arrival. In real world, this would be a new batch of data.
# df is passed to infer schema
synthetic_df = generate_synthetic_data(df, num_rows=100)
logger.info("Synthetic data generated.")

# Initialize DataProcessor
data_processor = DataProcessor(synthetic_df, config, spark)

# Preprocess the data
data_processor.preprocess()

# Split the data
X_train, X_test = data_processor.split_data()
logger.info("Training set shape: %s", X_train.shape)
logger.info("Test set shape: %s", X_test.shape)

# Save to catalog
logger.info("Saving data to catalog")
data_processor.save_to_catalog(X_train, X_test)

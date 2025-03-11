import argparse
import logging as logger

from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from power_consumption.config import ProjectConfig
from power_consumption.serving.model_serving import ModelServing

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

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

# Load project config
config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
logger.info("Loaded config file.")

catalog_name = config.catalog_name
schema_name = config.schema_name
endpoint_name = f"power-consumption-model-serving-{args.env}"

# Initialize model Serving Manager
model_serving = ModelServing(
    model_name=f"{config.catalog_name}.{config.schema_name}.power_consumption_model",
    endpoint_name=endpoint_name,
)

# Deploy the model serving endpoint
model_serving.deploy_or_update_serving_endpoint()
logger.info("Successfully deployed/updated the serving endpoint.")

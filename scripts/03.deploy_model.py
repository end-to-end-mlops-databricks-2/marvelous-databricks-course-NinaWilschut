import argparse

from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from src.power_consumption.config import ProjectConfig
from src.power_consumption.serving.model_serving import ModelServing

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
model_version = dbutils.jobs.taskValues.get(taskKey="train_model", key="model_version")

# Load project config
config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
logger.info("Loaded config file.")

catalog_name = config.catalog_name
schema_name = config.schema_name
endpoint_name = f"power-consumption-model-serving-{args.env}"

# Initialize Feature Lookup Serving Manager
model_server = ModelServing(
    model_name="power_consumption_model",
    endpoint_name=endpoint_name,
    table_name=f"{catalog_name}.{schema_name}.",
)

# Create the online table for house features
# model_server.create_online_table()
# logger.info("Created online table")

model_server.update_online_table(config=config)
logger.info("Updated online table")

# Deploy the model serving endpoint with feature lookup
model_server.deploy_or_update_serving_endpoint(version=model_version)
logger.info("Started deployment/update of the serving endpoint")

import argparse
import logging as logger

import mlflow
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from power_consumption.config import ProjectConfig, Tags
from power_consumption.models.model import Model

# Configure tracking uri
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

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

parser.add_argument(
    "--git_sha",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--job_run_id",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--branch",
    action="store",
    default=None,
    type=str,
    required=True,
)


args = parser.parse_args()
root_path = args.root_path
config_path = f"{root_path}/files/project_config.yml"

config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
tags_dict = {"git_sha": args.git_sha, "branch": args.branch, "job_run_id": args.job_run_id}
tags = Tags(**tags_dict)

# Initialize model
model = Model(config=config, tags=tags, spark=spark)
logger.info("Model initialized.")

# Load data
model.load_data()
logger.info("Data loaded.")

# Prepare features
model.prepare_features()
logger.info("Features prepared.")

# # Train and log model
# model.train()
# logger.info("Model trained.")
# model.log_model()
# logger.info("Model logged.")

# run_id = mlflow.search_runs(experiment_names=model.experiment_name, filter_string=f"tags.job_run_id = '{args.job_run_id}'").iloc[0].run_id
# logger.info(f"Run ID: {run_id}")

# model = mlflow.sklearn.load_model(f"runs:/{run_id}/lightgbm-pipeline-model")

# # Retrieve dataset for the current run
# model.retrieve_current_run_dataset()

# # Retrieve metadata for the current run
# model.retrieve_current_run_metadata()

# # Register model
# model.register_model()

# # Predict on the test set

# test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(10)

# X_test = test_set.drop(config.target).toPandas()

# predictions_df = model.load_latest_model_and_predict(X_test)

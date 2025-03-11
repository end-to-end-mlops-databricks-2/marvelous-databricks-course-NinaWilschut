import argparse
import logging as logger

import mlflow
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from power_consumption.config import ProjectConfig, Tags
from power_consumption.models.basic_model import BasicModel

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
basic_model = BasicModel(config=config, tags=tags, spark=spark)
logger.info("Model initialized.")

# Load data
basic_model.load_data()
basic_model.prepare_features()
logger.info("Data loaded.")

# Train the model
basic_model.train()
basic_model.log_model()
logger.info("Model training completed.")

run_id = mlflow.search_runs(
    experiment_names=["/Shared/power-consumption-nina"], filter_string=f"tags.branch='{args.branch}'"
).run_id[0]

loaded_model = mlflow.sklearn.load_model(f"runs:/{run_id}/lightgbm-pipeline-model")


# Retrieve dataset for the current run
basic_model.retrieve_current_run_dataset()


# Retrieve metadata for the current run
basic_model.retrieve_current_run_metadata()


# Register model
basic_model.register_model()

# Set the model_updated flag to 1
dbutils.jobs.taskValues.set(key="model_updated", value=1)

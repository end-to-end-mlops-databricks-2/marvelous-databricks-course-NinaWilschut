from pathlib import Path

import mlflow
from pyspark.sql import SparkSession

from power_consumption.config import ProjectConfig, Tags
from power_consumption.models.model import Model

# Default profile:
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# Profile called "course"
# mlflow.set_tracking_uri("databricks://course")
# mlflow.set_registry_uri("databricks-uc://course")
root = Path(__file__).parent.parent
config = ProjectConfig.from_yaml(config_path=root / "project_config.yml")

# config = ProjectConfig.from_yaml(config_path="../project_config.yml")
spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha": "abcd12345", "branch": "week2"})


# Initialize model with the config path
model = Model(config=config, tags=tags, spark=spark)

model.load_data()
model.prepare_features()

# Train + log the model (runs everything including MLflow logging)
model.train()
model.log_model()

run_id = mlflow.search_runs(
    experiment_names=["/Shared/power-consumption-nina"], filter_string="tags.branch='week2'"
).run_id[0]

model = mlflow.sklearn.load_model(f"runs:/{run_id}/lightgbm-pipeline-model")


# Retrieve dataset for the current run
model.retrieve_current_run_dataset()


# Retrieve metadata for the current run
model.retrieve_current_run_metadata()


# Register model
model.register_model()

# Predict on the test set

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(10)

X_test = test_set.drop(config.target).toPandas()

predictions_df = model.load_latest_model_and_predict(X_test)

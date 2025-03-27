import logging as logger

import mlflow
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient
from lightgbm import LGBMRegressor
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from power_consumption.config import ProjectConfig, Tags


class FeatureLookUpModel:
    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession):
        """
        Initialize the model with project configuration.
        """
        self.config = config
        self.spark = spark
        self.workspace = WorkspaceClient()
        self.fe = feature_engineering.FeatureEngineeringClient()

        # Extract settings from the config
        self.num_features = self.config.num_features
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name

        # Define table names and function name
        self.feature_table_name = f"{self.catalog_name}.{self.schema_name}.power_features"
        self.function_name = f"{self.catalog_name}.{self.schema_name}.calculate_percieved_temperature"

        # MLflow configuration
        self.experiment_name = self.config.experiment_name_fe
        self.tags = tags.dict()

    def create_feature_table(self):
        """
        Create or update the power_features table and populate it.
        """

        self.spark.sql(f"""
        CREATE OR REPLACE TABLE {self.feature_table_name}
        (Id STRING NOT NULL, datetime INT, general_diffuse_flows INT, diffuse_flows INT);
        """)
        self.spark.sql(f"ALTER TABLE {self.feature_table_name} ADD CONSTRAINT power_pk PRIMARY KEY(Id);")
        self.spark.sql(f"ALTER TABLE {self.feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

        self.spark.sql(
            f"INSERT INTO {self.feature_table_name} SELECT Id, datetime, general_diffuse_flows, diffuse_flows FROM {self.catalog_name}.{self.schema_name}.train_set"
        )
        self.spark.sql(
            f"INSERT INTO {self.feature_table_name} SELECT Id, datetime, general_diffuse_flows, diffuse_flows FROM {self.catalog_name}.{self.schema_name}.test_set"
        )
        logger.info("✅ Feature table created and populated.")

    def define_feature_function(self):
        """
        Define a function to calculate the perceived temperature.
        """
        self.spark.sql(f"""
        CREATE OR REPLACE FUNCTION {self.function_name}(temperature FLOAT, humidity FLOAT, wind_speed FLOAT)
        RETURNS FLOAT
        LANGUAGE PYTHON AS
        $$
        from datetime import datetime
        return temperature + 0.33 * (humidity - 100) + 0.7 * wind_speed
        $$
        """)
        logger.info("✅ Feature function defined.")

    def load_data(self):
        """
        Load training and testing data from Delta tables.
        """
        self.train_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set").drop(
            "temperature", "humidity", "wind_speed"
        )
        self.test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set").toPandas()

        self.train_set = self.train_set.withColumn("Id", self.train_set["Id"].cast("string"))

        logger.info("✅ Data successfully loaded.")

    def feature_engineering(self):
        """
        Perform feature engineering by linking data with feature tables.
        """
        self.training_set = self.fe.create_training_set(
            df=self.train_set,
            label=self.target,
            feature_lookups=[
                FeatureLookup(
                    table_name=self.feature_table_name,
                    feature_names=["datetime", "general_diffuse_flows", "diffuse_flows"],
                    lookup_key="Id",
                ),
                FeatureFunction(
                    udf_name=self.function_name,
                    output_name="perceived_temperature",
                    input_bindings={"temperature": "temperature", "humidity": "humidity", "wind_speed": "wind_speed"},
                ),
            ],
            exclude_columns=["update_timestamp_utc"],
        )

        self.training_df = self.training_set.load_df().toPandas()

        self.test_set["perceived_temperature"] = (
            self.test_set.temperature + 0.33 * (self.test_set.humidity - 100) + 0.7 * self.test_set.wind_speed
        )

        self.X_train = self.training_df[self.num_features + ["perceived_temperature"]]
        self.y_train = self.training_df[self.target]
        self.X_test = self.test_set[self.num_features + ["perceived_temperature"]]
        self.y_test = self.test_set[self.target]

        logger.info("✅ Feature engineering completed.")

    def train(self):
        """
        Train the model and log results to MLflow.
        """
        logger.info("🚀 Starting training...")

        pipeline = Pipeline(steps=[("regressor", LGBMRegressor(**self.parameters))])

        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            pipeline.fit(self.X_train, self.y_train)
            y_pred = pipeline.predict(self.X_test)

            mse = mean_squared_error(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)

            logger.info(f"📊 Mean Squared Error: {mse}")
            logger.info(f"📊 Mean Absolute Error: {mae}")
            logger.info(f"📊 R2 Score: {r2}")

            mlflow.log_param("model_type", "LightGBM with preprocessing")
            mlflow.log_params(self.parameters)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2_score", r2)
            signature = infer_signature(self.X_train, y_pred)

            self.fe.log_model(
                model=pipeline,
                flavor=mlflow.sklearn,
                artifact_path="lightgbm-pipeline-model-fe",
                training_set=self.training_set,
                signature=signature,
            )

    def register_model(self):
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/lightgbm-pipeline-model-fe",
            name=f"{self.catalog_name}.{self.schema_name}.power_consumption_model_fe",
            tags=self.tags,
        )

        # Fetch the latest version dynamically
        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=f"{self.catalog_name}.{self.schema_name}.power_consumption_model_fe",
            alias="latest-model",
            version=latest_version,
        )

        return latest_version

    def load_latest_model_and_predict(self, X: DataFrame):
        """
        Load the trained model from MLflow using Feature Engineering Client and make predictions.
        """
        model_uri = f"models:/{self.catalog_name}.{self.schema_name}.power_consumption_model_fe@latest-model"

        predictions = self.fe.score_batch(model_uri=model_uri, df=X)
        return predictions

    def update_feature_table(self):
        """
        Updates the house_features table with the latest records from train and test sets.
        """
        queries = [
            f"""
            WITH max_timestamp AS (
                SELECT MAX(update_timestamp_utc) AS max_update_timestamp
                FROM {self.config.catalog_name}.{self.config.schema_name}.train_set
            )
            INSERT INTO {self.feature_table_name}
            SELECT Id, datetime, general_diffuse_flows, diffuse_flows
            FROM {self.config.catalog_name}.{self.config.schema_name}.train_set
            WHERE update_timestamp_utc >= (SELECT max_update_timestamp FROM max_timestamp)
            """,
            f"""
            WITH max_timestamp AS (
                SELECT MAX(update_timestamp_utc) AS max_update_timestamp
                FROM {self.config.catalog_name}.{self.config.schema_name}.test_set
            )
            INSERT INTO {self.feature_table_name}
            SELECT Id, datetime, general_diffuse_flows, diffuse_flows
            FROM {self.config.catalog_name}.{self.config.schema_name}.test_set
            WHERE update_timestamp_utc >= (SELECT max_update_timestamp FROM max_timestamp)
            """,
        ]

        for query in queries:
            logger.info("Executing SQL update query...")
            self.spark.sql(query)
        logger.info("Power features table updated successfully.")

    def model_improved(self, test_set: DataFrame):
        """
        Evaluate the model performance on the test set.
        """
        X_test = test_set.drop(self.config.target)

        predictions_latest = self.load_latest_model_and_predict(X_test).withColumnRenamed(
            "prediction", "prediction_latest"
        )

        current_model_uri = f"runs:/{self.run_id}/lightgbm-pipeline-model-fe"
        predictions_current = self.fe.score_batch(model_uri=current_model_uri, df=X_test).withColumnRenamed(
            "prediction", "prediction_current"
        )

        test_set = test_set.select("Id", "zone_1_power_consumption")

        logger.info("Predictions are ready.")

        # Join the DataFrames on the 'id' column
        df = test_set.join(predictions_current, on="Id").join(predictions_latest, on="Id")

        # Calculate the absolute error for each model
        df = df.withColumn("error_current", F.abs(df["zone_1_power_consumption"] - df["prediction_current"]))
        df = df.withColumn("error_latest", F.abs(df["zone_1_power_consumption"] - df["prediction_latest"]))

        # Calculate the Mean Absolute Error (MAE) for each model
        mae_current = df.agg(F.mean("error_current")).collect()[0][0]
        mae_latest = df.agg(F.mean("error_latest")).collect()[0][0]

        # Compare models based on MAE
        logger.info(f"MAE for Current Model: {mae_current}")
        logger.info(f"MAE for Latest Model: {mae_latest}")

        if mae_current < mae_latest:
            logger.info("Current Model performs better. Registering new model.")
            return True
        else:
            logger.info("New Model performs worse. Keeping the old model.")
            return False

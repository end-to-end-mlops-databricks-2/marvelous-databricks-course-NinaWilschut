import time

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

from src.power_consumption.config import ProjectConfig


class DataProcessor:
    def __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig, spark: SparkSession):
        self.df = pandas_df  # Store the DataFrame as self.df
        self.config = config  # Store the configuration
        self.spark = spark

    def preprocess(self):
        """Preprocess the DataFrame stored in self.df"""
        # Remove extra target columns Zone 2 Power Consumption and Zone 3 Power Consumption (the last two)
        # drop the last two columns
        self.df = self.df.iloc[:, :-2]

        # Rename columns to convert spaces to underscores and make all lowercase
        self.df.columns = [col.lower().replace(" ", "_") for col in self.df.columns]

        # Handle numeric features
        num_features = self.config.num_features
        for col in num_features:
            if col != "datetime":
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
            else:
                self.df[col] = pd.to_datetime(self.df[col], errors="coerce").astype("int64") / 10**9

        # Add id column
        self.df["id"] = self.df.index

        # Extract target and relevant features
        target = self.config.target
        relevant_columns = num_features + [target] + ["id"]
        self.df = self.df[relevant_columns]
        self.df["id"] = self.df["id"].astype("str")

    def split_data(self, test_size=0.2, random_state=42):
        """Split the DataFrame (self.df) into training and test sets."""
        train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train_set, test_set

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame):
        """Save the train and test sets into Databricks tables."""

        train_set_with_timestamp = self.spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = self.spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        )

        test_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        )

    def enable_change_data_feed(self):
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )


def generate_synthetic_data(df, num_rows=10):
    """Generates synthetic data based on the distribution of the input DataFrame."""
    synthetic_data = pd.DataFrame()

    for column in df.columns:
        if column == "id":
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            synthetic_data[column] = np.random.normal(df[column].mean(), df[column].std(), num_rows)

        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            min_date, max_date = df[column].min(), df[column].max()
            date_range = pd.date_range(start=min_date, end=max_date, periods=num_rows)
            synthetic_data[column] = np.random.choice(date_range, num_rows)

        else:
            synthetic_data[column] = np.random.choice(df[column], num_rows)

    # Convert relevant numeric columns to integers
    int_columns = {
        "temperature",
        "humidity",
        "wind_speed",
        "general_diffuse_flows",
        "diffuse_flows",
    }
    for col in int_columns.intersection(df.columns):
        synthetic_data[col] = pd.to_numeric(synthetic_data[col], errors="coerce")

    timestamp_base = int(time.time() * 1000)
    synthetic_data["id"] = [str(timestamp_base + i) for i in range(num_rows)]

    return synthetic_data

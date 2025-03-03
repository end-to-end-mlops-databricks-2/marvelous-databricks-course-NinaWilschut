import logging

import pandas as pd
import yaml
from pyspark.sql import SparkSession

from src.power_consumption.config import ProjectConfig

# from pyspark.sql import SparkSession
# from pyspark.sql.functions import current_timestamp, to_utc_timestamp
# from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

config = ProjectConfig.from_yaml(config_path="../project_config.yml")

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))


catalog_name = config["catalog_name"]
schema_name = config["schema_name"]


# Load the house prices dataset
spark = SparkSession.builder.getOrCreate()

df = spark.read.csv(
    f"/Volumes/{config.catalog_name}/{config.schema_name}/data/data.csv", header=True, inferSchema=True
).toPandas()


# Handle numeric features
num_features = config["num_features"]
for col in num_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Extract target and relevant features
target = config["target"]
relevant_columns = num_features + [target] + ["Id"]
print(relevant_columns)
df = df[relevant_columns]
df["Id"] = df["Id"].astype("str")

print(df.dtypes)
print(df.describe())
print(df)


# train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

# train_set_with_timestamp = spark.createDataFrame(train_set).withColumn(
#     "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
# )

# test_set_with_timestamp = spark.createDataFrame(test_set).withColumn(
#     "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
# )

# train_set_with_timestamp.write.mode("append").saveAsTable(f"{catalog_name}.{schema_name}.train_set")

# test_set_with_timestamp.write.mode("append").saveAsTable(f"{catalog_name}.{schema_name}.test_set")

# spark.sql(f"ALTER TABLE {catalog_name}.{schema_name}.train_set SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

# spark.sql(f"ALTER TABLE {catalog_name}.{schema_name}.test_set SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

import logging
from pathlib import Path

import pandas as pd
import yaml

# from pyspark.sql import SparkSession
from databricks.connect import DatabricksSession

# from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

from power_consumption.config import ProjectConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

root = Path(__file__).parent.parent
config = ProjectConfig.from_yaml(config_path=root / "project_config.yml")

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))


# Load the house prices dataset
# spark = SparkSession.builder.getOrC
spark = DatabricksSession.builder.getOrCreate()

df = spark.read.csv(
    f"/Volumes/{config.catalog_name}/{config.schema_name}/data/data.csv", header=True, inferSchema=True
).toPandas()


# Remove extra target columns Zone 2 Power Consumption and Zone 3 Power Consumption (the last two)
# drop the last two columns
df = df.iloc[:, :-2]

# Rename columns to convert spaces to underscores and make all lowercase
df.columns = [col.lower().replace(" ", "_") for col in df.columns]

# Handle numeric features
num_features = config.num_features
for col in num_features:
    if col == "datetime":
        df[col] = pd.to_datetime(df[col], errors="coerce").astype("int64") / 10**9
    else:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Add id column
df["id"] = df.index

# Extract target and relevant features
target = config.target
relevant_columns = num_features + [target] + ["id"]
df = df[relevant_columns]
df["id"] = df["id"].astype("str")

train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

train_set_with_timestamp = spark.createDataFrame(train_set).withColumn(
    "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
)

test_set_with_timestamp = spark.createDataFrame(test_set).withColumn(
    "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
)

train_set_with_timestamp.write.mode("append").saveAsTable(f"{config.catalog_name}.{config.schema_name}.train_set")

test_set_with_timestamp.write.mode("append").saveAsTable(f"{config.catalog_name}.{config.schema_name}.test_set")

spark.sql(
    f"ALTER TABLE {config.catalog_name}.{config.schema_name}.train_set SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
)

spark.sql(
    f"ALTER TABLE {config.catalog_name}.{config.schema_name}.test_set SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
)
